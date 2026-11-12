"""
Módulo de Renderização com Leonardo.ai API - RenderX (Fase 3)
Responsável por:
1. Fazer upload da imagem original (rascunho/print 3D) via endpoint init-image.
2. Disparar a geração fotorrealista via Image-to-Image (Init Image) com SDXL.
3. Fazer polling até a conclusão da renderização.
4. Fazer o download da imagem final e salvá-la no diretório output/.
"""

import os
import time
import json
import uuid
from pathlib import Path
from typing import Union, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

LEONARDO_BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"

# Modelo base SDXL 1.0 na API do Leonardo.ai (sdVersion SDXL_1_0).
# O ID anterior (b7aa9939-...) era resolvido pela API como sdVersion v1_5,
# o que causava o conflito com preprocessors SDXL.
DEFAULT_MODEL_ID = "16e7060a-803e-4df3-97ee-edcfa5dc9cc8"

# preprocessorId do ControlNet "Edge to Image" para SDXL (SD 1.5 usa 1; SDXL usa 19).
# SDXL não possui "Sketch to Image", então PAPER_SKETCH também usa Edge to Image.
SDXL_EDGE_PREPROCESSOR_ID = 19
CONTROLNET_PREPROCESSORS = {
    "DIGITAL_3D": SDXL_EDGE_PREPROCESSOR_ID,
    "PAPER_SKETCH": SDXL_EDGE_PREPROCESSOR_ID,
}
VALID_STRENGTH_TYPES = ("Low", "Mid", "High", "Ultra", "Max")


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
        "authorization": f"Bearer {api_key}"
    }


def upload_init_image(
    image_input: Union[str, Path],
    api_key: Optional[str] = None
) -> str:
    """
    Realiza o upload da imagem de entrada para o bucket de init-image da Leonardo.ai.

    Args:
        image_input: Caminho do arquivo da imagem local (str ou Path).
        api_key: Chave da API do Leonardo.ai (opcional).

    Returns:
        ID da imagem carregada (init_image_id).

    Raises:
        FileNotFoundError: Se a imagem informada não existir.
        RuntimeError: Se o upload ou a obtenção da URL pré-assinada falhar.
    """
    key = _get_api_key(api_key)
    path = Path(image_input)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {path}")

    # Determina a extensão do arquivo
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in ["jpg", "png", "webp"]:
        ext = "jpg"

    # 1. Solicita URL pré-assinada para upload da init-image
    url_endpoint = f"{LEONARDO_BASE_URL}/init-image"
    headers = _get_headers(key)
    payload = {"extension": ext}

    try:
        response = requests.post(url_endpoint, headers=headers, json=payload, timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de conexão com o endpoint init-image: {e}") from e

    if response.status_code != 200:
        if response.status_code == 401:
            raise PermissionError(
                "Falha de autorização (401) no Leonardo.ai. "
                "Verifique sua LEONARDO_API_KEY no arquivo .env e certifique-se de que sua conta "
                "possui créditos de API ativos na plataforma (https://cloud.leonardo.ai)."
            )
        raise RuntimeError(
            f"Erro ao obter URL de upload (HTTP {response.status_code}): {response.text}"
        )

    res_json = response.json()
    init_data = res_json.get("uploadInitImage")
    if not init_data:
        raise RuntimeError(f"Resposta inválida de init-image: {res_json}")

    image_id = init_data["id"]
    upload_url = init_data["url"]
    fields_raw = init_data.get("fields", "{}")
    fields = json.loads(fields_raw) if isinstance(fields_raw, str) else fields_raw

    # 2. Faz o upload da imagem para o S3 via multipart/form-data
    try:
        with open(path, "rb") as img_file:
            files = {"file": (path.name, img_file, f"image/{ext}")}
            s3_response = requests.post(upload_url, data=fields, files=files, timeout=40)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro no upload da imagem para o bucket S3: {e}") from e

    if s3_response.status_code not in [200, 204]:
        raise RuntimeError(
            f"Upload para o storage S3 falhou (HTTP {s3_response.status_code}): {s3_response.text}"
        )

    return image_id


def create_generation_job(
    prompt: str,
    init_image_id: str,
    input_type: str = "DIGITAL_3D",
    strength_type: str = "High",
    model_id: str = DEFAULT_MODEL_ID,
    api_key: Optional[str] = None
) -> str:
    """
    Submete uma geração com ControlNet (Image Guidance) em SDXL no Leonardo.ai.

    Args:
        prompt: Prompt descritivo em inglês (sd_prompt da Fase 2).
        init_image_id: ID retornado pelo upload da init-image.
        input_type: "DIGITAL_3D" ou "PAPER_SKETCH" (define o preprocessor).
        strength_type: Força do ControlNet: Low, Mid, High, Ultra ou Max.
        model_id: ID do modelo SDXL (padrão: SDXL 1.0).
        api_key: Chave da API do Leonardo.ai (opcional).

    Returns:
        generation_id do job criado.
    """
    if input_type not in CONTROLNET_PREPROCESSORS:
        raise ValueError(f"input_type inválido: {input_type}. Use {list(CONTROLNET_PREPROCESSORS)}.")
    if strength_type not in VALID_STRENGTH_TYPES:
        raise ValueError(f"strength_type inválido: {strength_type}. Use {VALID_STRENGTH_TYPES}.")

    key = _get_api_key(api_key)
    headers = _get_headers(key)
    endpoint = f"{LEONARDO_BASE_URL}/generations"

    # ControlNet em SDXL: tudo encapsulado em "controlnets", sem init_image_id /
    # init_strength / weight na raiz. modelId e preprocessorId são ambos SDXL.
    payload: Dict[str, Any] = {
        "prompt": prompt,
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

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=25)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de conexão com o endpoint de geração: {e}") from e

    # Tratamento específico de respostas HTTP
    if response.status_code == 401:
        raise PermissionError(
            "Falha de autenticação (401 Unauthorized) no Leonardo.ai: "
            "Acesse https://cloud.leonardo.ai e verifique sua assinatura e saldo de créditos da API."
        )
    elif response.status_code == 402:
        raise RuntimeError(
            "Créditos de API insuficientes (402 Payment Required): Seus créditos da Leonardo API acabaram."
        )
    elif response.status_code == 429:
        raise RuntimeError(
            "Limite de requisições excedido (429 Rate Limit): Aguarde alguns instantes antes de tentar novamente."
        )
    elif response.status_code != 200:
        raise RuntimeError(
            f"Erro na requisição de geração (HTTP {response.status_code}): {response.text}"
        )

    res_data = response.json()
    generation_job = res_data.get("sdGenerationJob")
    if not generation_job or "generationId" not in generation_job:
        raise RuntimeError(f"Resposta inesperada ao criar geração: {res_data}")

    return generation_job["generationId"]


def poll_generation_result(
    generation_id: str,
    api_key: Optional[str] = None,
    timeout_seconds: int = 120,
    poll_interval: int = 4
) -> str:
    """
    Executa o polling periódico no Leonardo.ai até que a imagem esteja pronta.

    Args:
        generation_id: ID da geração.
        api_key: Chave da API.
        timeout_seconds: Tempo máximo de espera em segundos (padrão: 120s).
        poll_interval: Intervalo entre verificações (padrão: 4s).

    Returns:
        URL pública da imagem renderizada.

    Raises:
        TimeoutError: Se o tempo limite de espera for atingido.
        RuntimeError: Se a geração falhar na API.
    """
    key = _get_api_key(api_key)
    headers = _get_headers(key)
    endpoint = f"{LEONARDO_BASE_URL}/generations/{generation_id}"

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            res = requests.get(endpoint, headers=headers, timeout=15)
        except requests.exceptions.RequestException:
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

    raise TimeoutError(f"Tempo limite excedido ({timeout_seconds}s) aguardando a renderização do Leonardo.ai.")


def download_rendered_image(
    image_url: str,
    output_dir: Union[str, Path] = "output",
    filename_prefix: str = "render"
) -> Path:
    """
    Faz o download da imagem renderizada a partir da URL e a salva no diretório output.

    Args:
        image_url: URL pública da imagem gerada.
        output_dir: Diretório de destino (criado automaticamente caso não exista).
        filename_prefix: Prefixo do arquivo salvo.

    Returns:
        Path do arquivo salvo localmente.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    target_file = out_path / f"{filename_prefix}_{unique_suffix}.jpg"

    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(target_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro ao baixar a imagem renderizada de {image_url}: {e}") from e

    return target_file


def render_carpentry_design(
    image_path: Union[str, Path],
    vision_data: Union[Dict[str, Any], str],
    api_key: Optional[str] = None,
    output_dir: Union[str, Path] = "output",
    input_type: Optional[str] = None,
    strength_type: str = "High",
    model_id: str = DEFAULT_MODEL_ID
) -> Path:
    """
    Função principal integrada da Fase 3:
    Recebe a imagem original e os dados estruturados da Fase 2 (ou sd_prompt),
    faz o upload via presigned S3, dispara a renderização Image-to-Image clássica no Leonardo.ai,
    aguarda a conclusão via polling e salva o resultado final no diretório output/.

    Args:
        image_path: Caminho da imagem original (rascunho ou print 3D).
        vision_data: Dict com a chave 'sd_prompt', ou string com o prompt.
        api_key: Chave de API do Leonardo.ai (opcional).
        output_dir: Pasta de destino da imagem gerada (padrão: "output").
        input_type: "DIGITAL_3D" ou "PAPER_SKETCH"; se None, usa vision_data["input_type"] (fallback DIGITAL_3D).
        strength_type: Força do ControlNet (Low, Mid, High, Ultra, Max).
        model_id: ID do modelo base SDXL no Leonardo.ai.

    Returns:
        Path do arquivo final fotorrealista renderizado.
    """
    # 1. Extração do prompt
    if isinstance(vision_data, dict):
        prompt = vision_data.get("sd_prompt", "")
    else:
        prompt = str(vision_data)

    if input_type is None:
        input_type = vision_data.get("input_type", "DIGITAL_3D") if isinstance(vision_data, dict) else "DIGITAL_3D"
    if input_type not in CONTROLNET_PREPROCESSORS:
        input_type = "DIGITAL_3D"

    if not prompt.strip():
        raise ValueError("O prompt para renderização (sd_prompt) não pode ser vazio.")

    print(f"[*] [1/4] Fazendo upload da imagem original ({image_path})...")
    init_image_id = upload_init_image(image_path, api_key=api_key)
    print(f"    -> Imagem enviada com sucesso! ID: {init_image_id}")

    print(f"[*] [2/4] Disparando renderização no Leonardo.ai (Model: {model_id} | ControlNet: {input_type}/{strength_type})...")
    generation_id = create_generation_job(
        prompt=prompt,
        init_image_id=init_image_id,
        input_type=input_type,
        strength_type=strength_type,
        model_id=model_id,
        api_key=api_key
    )
    print(f"    -> Job de geração criado! ID: {generation_id}")

    print(f"[*] [3/4] Aguardando processamento da imagem pela IA (polling)...")
    image_url = poll_generation_result(generation_id, api_key=api_key)
    print(f"    -> Renderização concluída! URL: {image_url}")

    print(f"[*] [4/4] Baixando imagem final para o diretório '{output_dir}'...")
    saved_path = download_rendered_image(image_url, output_dir=output_dir)
    print(f"    -> Imagem salva com sucesso em: {saved_path}")

    return saved_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso para teste manual:")
        print("  python render_module.py <caminho_imagem_original> <prompt_ou_json>")
        sys.exit(1)

    img_path = sys.argv[1]
    prompt_arg = sys.argv[2]

    try:
        data = json.loads(prompt_arg)
    except json.JSONDecodeError:
        data = prompt_arg

    try:
        final_file = render_carpentry_design(img_path, data)
        print(f"\n[+] Renderização concluída com sucesso: {final_file}")
    except Exception as err:
        print(f"\n[-] Erro durante processo de renderização: {err}")
        sys.exit(1)
