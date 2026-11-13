"""
Módulo de Renderização com Leonardo.ai API - RenderX (Fase 3 - Refatorado)
Responsável por:
1. Isolar cada requisição em uma pasta de job única (`outputs/<job_id>/`) para observabilidade.
2. Fazer upload da imagem original via endpoint `init-image` (com retry/backoff e timeout).
3. Disparar a geração fotorrealista via SDXL + ControlNet (`preprocessorId: 19`, `strengthType: High`)
   recebendo `sd_prompt` e `negative_prompt` gerados programaticamente na Fase 2.
4. Fazer polling resiliente até a conclusão da renderização.
5. Salvar a imagem original, o JSON do Pydantic (`spec.json`), metadados e a imagem final em `outputs/<job_id>/`.
"""

import os
import time
import json
import uuid
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Dict, Any, Optional
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Carrega variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LEONARDO_BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"

# Modelo base SDXL 1.0 na API do Leonardo.ai (sdVersion SDXL_1_0).
DEFAULT_MODEL_ID = "16e7060a-803e-4df3-97ee-edcfa5dc9cc8"

# preprocessorId do ControlNet "Edge to Image" para SDXL (19).
SDXL_EDGE_PREPROCESSOR_ID = 19
CONTROLNET_PREPROCESSORS = {
    "DIGITAL_3D": SDXL_EDGE_PREPROCESSOR_ID,
    "PAPER_SKETCH": SDXL_EDGE_PREPROCESSOR_ID,
}
VALID_STRENGTH_TYPES = ("Low", "Mid", "High", "Ultra", "Max")

# Timeouts padrão (segundos)
DEFAULT_API_TIMEOUT = 25
UPLOAD_S3_TIMEOUT = 45
DOWNLOAD_IMAGE_TIMEOUT = 40


class TransientLeonardoError(RuntimeError):
    """Erro transitório (HTTP 429, 5xx ou falha de rede) elegível para retry com backoff."""


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Valida e retorna a chave da API do Leonardo.ai."""
    key = api_key or os.getenv("LEONARDO_API_KEY")
    if not key or key == "sua_chave_leonardo_aqui":
        raise ValueError(
            "LEONARDO_API_KEY não configurada. Defina no arquivo .env ou forneça como parâmetro."
        )
    return key.strip()


def _get_headers(api_key: str) -> Dict[str, str]:
    """Retorna os headers HTTP obrigatórios para autenticação na Leonardo.ai API."""
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }


def _raise_for_leonardo_status(response: requests.Response, context: str) -> None:
    """
    Classifica erros HTTP da API do Leonardo.ai entre falhas permanentes (4xx)
    e falhas transitórias elegíveis para retry (429, 5xx).
    """
    code = response.status_code
    if code == 200:
        return
    if code == 401:
        raise PermissionError(
            f"Falha de autenticação (401 Unauthorized) em {context}: "
            "Verifique sua LEONARDO_API_KEY e créditos ativos em https://cloud.leonardo.ai."
        )
    if code == 402:
        raise RuntimeError(
            f"Créditos insuficientes (402 Payment Required) em {context}: "
            "Seus créditos da Leonardo API acabaram."
        )
    if code == 429 or 500 <= code < 600:
        raise TransientLeonardoError(
            f"Erro temporário do servidor Leonardo.ai em {context} (HTTP {code}): {response.text}"
        )
    raise RuntimeError(
        f"Erro permanente em {context} (HTTP {code}): {response.text}"
    )


@retry(
    retry=retry_if_exception_type((TransientLeonardoError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _request_with_retry(method: str, url: str, timeout: int = DEFAULT_API_TIMEOUT, **kwargs) -> requests.Response:
    """Executa uma requisição HTTP com timeout explícito e retry exponencial para falhas transitórias."""
    response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    if response.status_code == 429 or 500 <= response.status_code < 600:
        raise TransientLeonardoError(
            f"HTTP {response.status_code} em {url}: {response.text[:300]}"
        )
    return response


def generate_job_id() -> str:
    """Gera um identificador único legível (timestamp UTC + sufixo UUID) para o job."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}_{short_uuid}"


def _save_job_artifacts(
    job_dir: Path,
    source_image: Path,
    sd_prompt: str,
    negative_prompt: str,
    spec_data: Optional[Any] = None,
) -> None:
    """Salva a imagem original e o JSON estruturado (Pydantic/dict) na pasta do job."""
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cópia da imagem original enviada
    ext = source_image.suffix.lower() or ".jpg"
    shutil.copy2(source_image, job_dir / f"original{ext}")

    # 2. JSON gerado pelo Pydantic (se fornecido)
    if spec_data is not None:
        spec_file = job_dir / "spec.json"
        if hasattr(spec_data, "model_dump_json"):
            spec_file.write_text(spec_data.model_dump_json(indent=2), encoding="utf-8")
        elif isinstance(spec_data, dict):
            spec_file.write_text(json.dumps(spec_data, indent=2, ensure_ascii=False), encoding="utf-8")
        elif isinstance(spec_data, str):
            spec_file.write_text(spec_data, encoding="utf-8")

    # 3. Registro inicial dos prompts
    prompts_file = job_dir / "prompts.json"
    prompts_file.write_text(
        json.dumps(
            {"sd_prompt": sd_prompt, "negative_prompt": negative_prompt},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def upload_init_image(
    image_input: Union[str, Path],
    api_key: Optional[str] = None,
) -> str:
    """
    Realiza o upload da imagem de entrada para o bucket de init-image da Leonardo.ai
    com timeout explícito e retry exponencial para erros transitórios.
    """
    key = _get_api_key(api_key)
    path = Path(image_input)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {path}")

    ext = path.suffix.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in ("jpg", "png", "webp"):
        ext = "jpg"

    # 1. Solicita URL pré-assinada
    url_endpoint = f"{LEONARDO_BASE_URL}/init-image"
    headers = _get_headers(key)
    payload = {"extension": ext}

    response = _request_with_retry(
        "POST",
        url_endpoint,
        timeout=DEFAULT_API_TIMEOUT,
        headers=headers,
        json=payload,
    )
    _raise_for_leonardo_status(response, "init-image")

    res_json = response.json()
    init_data = res_json.get("uploadInitImage")
    if not init_data:
        raise RuntimeError(f"Resposta inválida de init-image: {res_json}")

    image_id = init_data["id"]
    upload_url = init_data["url"]
    fields_raw = init_data.get("fields", "{}")
    fields = json.loads(fields_raw) if isinstance(fields_raw, str) else fields_raw

    # 2. Faz o upload da imagem para o bucket S3 via multipart/form-data (com retry)
    @retry(
        retry=retry_if_exception_type((TransientLeonardoError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _upload_to_s3() -> requests.Response:
        with open(path, "rb") as img_file:
            files = {"file": (path.name, img_file, f"image/{ext}")}
            s3_res = requests.post(upload_url, data=fields, files=files, timeout=UPLOAD_S3_TIMEOUT)
        if s3_res.status_code == 429 or 500 <= s3_res.status_code < 600:
            raise TransientLeonardoError(f"S3 upload HTTP {s3_res.status_code}")
        return s3_res

    s3_response = _upload_to_s3()
    if s3_response.status_code not in (200, 204):
        raise RuntimeError(
            f"Upload para o storage S3 falhou (HTTP {s3_response.status_code}): {s3_response.text}"
        )

    return image_id


def create_generation_job(
    sd_prompt: str,
    init_image_id: str,
    negative_prompt: str = "",
    input_type: str = "DIGITAL_3D",
    strength_type: str = "High",
    model_id: str = DEFAULT_MODEL_ID,
    api_key: Optional[str] = None,
) -> str:
    """
    Submete uma geração com ControlNet (Edge to Image XL - preprocessorId 19)
    em SDXL 1.0 no Leonardo.ai mantendo a estrutura de payload validada.
    """
    if input_type not in CONTROLNET_PREPROCESSORS:
        raise ValueError(f"input_type inválido: {input_type}. Use {list(CONTROLNET_PREPROCESSORS)}.")
    if strength_type not in VALID_STRENGTH_TYPES:
        raise ValueError(f"strength_type inválido: {strength_type}. Use {VALID_STRENGTH_TYPES}.")

    key = _get_api_key(api_key)
    headers = _get_headers(key)
    endpoint = f"{LEONARDO_BASE_URL}/generations"

    # Estrutura exata validada na Fase 1.5/3: ControlNet encapsulado em array "controlnets"
    payload: Dict[str, Any] = {
        "prompt": sd_prompt,
        "modelId": model_id,
        "width": 1024,
        "height": 768,
        "num_images": 1,
        "controlnets": [
            {
                "initImageId": init_image_id,
                "initImageType": "UPLOADED",
                "preprocessorId": CONTROLNET_PREPROCESSORS[input_type],
                "weight": 1.0,
                "strengthType": strength_type,
            }
        ],
    }

    if negative_prompt and negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt.strip()

    response = _request_with_retry(
        "POST",
        endpoint,
        timeout=DEFAULT_API_TIMEOUT,
        headers=headers,
        json=payload,
    )
    _raise_for_leonardo_status(response, "generations")

    res_data = response.json()
    generation_job = res_data.get("sdGenerationJob")
    if not generation_job or "generationId" not in generation_job:
        raise RuntimeError(f"Resposta inesperada ao criar geração: {res_data}")

    return generation_job["generationId"]


def poll_generation_result(
    generation_id: str,
    api_key: Optional[str] = None,
    timeout_seconds: int = 120,
    poll_interval: int = 4,
) -> str:
    """
    Executa o polling periódico no Leonardo.ai até que a imagem esteja pronta,
    com tratamento de erros transitórios e timeout por requisição.
    """
    key = _get_api_key(api_key)
    headers = _get_headers(key)
    endpoint = f"{LEONARDO_BASE_URL}/generations/{generation_id}"

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            res = _request_with_retry("GET", endpoint, timeout=15, headers=headers)
        except (TransientLeonardoError, requests.exceptions.RequestException) as e:
            logger.warning("Falha transitória durante polling (%s): %s", generation_id, e)
            time.sleep(poll_interval)
            continue

        if res.status_code == 200:
            data = res.json()
            gen_data = data.get("generations_by_pk")
            if gen_data:
                status = gen_data.get("status")
                if status == "COMPLETE":
                    images = gen_data.get("generated_images", [])
                    if images and "url" in images[0]:
                        return images[0]["url"]
                    raise RuntimeError("Geração concluída, mas nenhuma imagem foi retornada.")
                elif status == "FAILED":
                    raise RuntimeError(f"A geração no Leonardo.ai falhou: {gen_data}")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Tempo limite excedido ({timeout_seconds}s) aguardando a renderização do Leonardo.ai."
    )


def download_rendered_image(
    image_url: str,
    target_file: Union[str, Path],
) -> Path:
    """
    Faz o download da imagem renderizada a partir da URL com retry e timeout explícito,
    salvando no caminho de destino informado.
    """
    dest = Path(target_file)
    dest.parent.mkdir(parents=True, exist_ok=True)

    @retry(
        retry=retry_if_exception_type((TransientLeonardoError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _download() -> None:
        response = requests.get(image_url, stream=True, timeout=DOWNLOAD_IMAGE_TIMEOUT)
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise TransientLeonardoError(f"HTTP {response.status_code} ao baixar imagem")
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    _download()
    return dest


def render_carpentry_design(
    image_path: Union[str, Path],
    sd_prompt: str,
    negative_prompt: str = "",
    spec_data: Optional[Any] = None,
    input_type: str = "DIGITAL_3D",
    strength_type: str = "High",
    model_id: str = DEFAULT_MODEL_ID,
    api_key: Optional[str] = None,
    outputs_base_dir: Union[str, Path] = "outputs",
    job_id: Optional[str] = None,
) -> Path:
    """
    Função principal integrada da Fase 3 (Refatorada):
    Recebe as strings finais (`sd_prompt` e `negative_prompt`) geradas pelo `build_prompt`
    da Fase 2, isola os artefatos em `outputs/<job_id>/`, executa o render SDXL com ControlNet
    no Leonardo.ai com retries e timeouts explícitos, e salva a imagem final no diretório do job.

    Args:
        image_path: Caminho da imagem original (rascunho ou print 3D).
        sd_prompt: Prompt positivo final gerado pelo `build_prompt`.
        negative_prompt: Prompt negativo final gerado pelo `build_negative_prompt`.
        spec_data: Instância Pydantic `FurnitureSpec` (ou dict/JSON) para log em `spec.json`.
        input_type: "DIGITAL_3D" ou "PAPER_SKETCH".
        strength_type: Força do ControlNet (Low, Mid, High, Ultra, Max).
        model_id: ID do modelo base SDXL 1.0 no Leonardo.ai.
        api_key: Chave de API do Leonardo.ai (opcional).
        outputs_base_dir: Diretório raiz de outputs (padrão: "outputs").
        job_id: ID customizado do job (gerado automaticamente se omitido).

    Returns:
        Path do arquivo final renderizado (`outputs/<job_id>/render.jpg`).
    """
    src_path = Path(image_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Imagem original não encontrada: {src_path}")

    if not sd_prompt or not sd_prompt.strip():
        raise ValueError("O parâmetro `sd_prompt` não pode ser vazio.")

    if input_type not in CONTROLNET_PREPROCESSORS:
        input_type = "DIGITAL_3D"

    active_job_id = job_id or generate_job_id()
    job_dir = Path(outputs_base_dir) / active_job_id

    logger.info("[%s] Criando diretório do job em: %s", active_job_id, job_dir)
    _save_job_artifacts(
        job_dir=job_dir,
        source_image=src_path,
        sd_prompt=sd_prompt.strip(),
        negative_prompt=(negative_prompt or "").strip(),
        spec_data=spec_data,
    )

    logger.info("[%s] [1/4] Fazendo upload da imagem original (%s)...", active_job_id, src_path.name)
    init_image_id = upload_init_image(src_path, api_key=api_key)
    logger.info("[%s] -> Init Image ID: %s", active_job_id, init_image_id)

    logger.info(
        "[%s] [2/4] Disparando renderização SDXL + ControlNet (Model: %s | Input: %s | Strength: %s)...",
        active_job_id,
        model_id,
        input_type,
        strength_type,
    )
    generation_id = create_generation_job(
        sd_prompt=sd_prompt.strip(),
        negative_prompt=(negative_prompt or "").strip(),
        init_image_id=init_image_id,
        input_type=input_type,
        strength_type=strength_type,
        model_id=model_id,
        api_key=api_key,
    )
    logger.info("[%s] -> Generation ID: %s", active_job_id, generation_id)

    logger.info("[%s] [3/4] Aguardando processamento da imagem (polling)...", active_job_id)
    image_url = poll_generation_result(generation_id, api_key=api_key)
    logger.info("[%s] -> Renderização concluída! URL: %s", active_job_id, image_url)

    final_image_path = job_dir / "render.jpg"
    logger.info("[%s] [4/4] Baixando imagem final para: %s", active_job_id, final_image_path)
    saved_path = download_rendered_image(image_url, target_file=final_image_path)

    # Salva metadados completos de execução para observabilidade
    metadata = {
        "job_id": active_job_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "input_type": input_type,
        "strength_type": strength_type,
        "init_image_id": init_image_id,
        "generation_id": generation_id,
        "image_url": image_url,
        "rendered_file": str(saved_path.name),
    }
    (job_dir / "job_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("[%s] Pipeline de renderização finalizado com sucesso!", active_job_id)
    return saved_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso para teste manual:")
        print("  python render_module.py <caminho_imagem_original> <sd_prompt> [negative_prompt]")
        sys.exit(1)

    img_path_arg = sys.argv[1]
    sd_prompt_arg = sys.argv[2]
    neg_prompt_arg = sys.argv[3] if len(sys.argv) > 3 else ""

    try:
        final_file = render_carpentry_design(
            image_path=img_path_arg,
            sd_prompt=sd_prompt_arg,
            negative_prompt=neg_prompt_arg,
        )
        print(f"\n[+] Renderização concluída com sucesso: {final_file}")
    except Exception as err:
        print(f"\n[-] Erro durante processo de renderização: {err}")
        sys.exit(1)
