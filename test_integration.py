"""
Script de Teste de Integração Ponta a Ponta (E2E) - RenderX (Fases 2 + 2.5 + 3)
Executa o pipeline completo com chamada real ao Google Gemini (Visão/OCR),
carregamento de estilo YAML e simulação (Mock) das chamadas HTTP para a API do Leonardo.ai,
validando a criação dos artefatos de observabilidade em `outputs/<job_id>/`.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from vision_module import (
    FurnitureSpec,
    analyze_carpentry_image,
    build_prompt,
    build_negative_prompt,
)
from style_loader import load_style
from render_module import render_carpentry_design


def _create_mock_gray_jpeg_bytes(width: int = 1024, height: int = 768) -> bytes:
    """Gera os bytes de uma imagem cinza simples via Pillow para simular o render.jpg."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _mock_requests_request(method: str, url: str, **kwargs) -> MagicMock:
    """Intercepta `requests.request` usado por `_request_with_retry` no render_module."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    if method == "POST" and url.endswith("/init-image"):
        mock_resp.json.return_value = {
            "uploadInitImage": {
                "id": "mock-init-img-12345",
                "url": "https://mock-s3-bucket.local/upload",
                "fields": '{"key": "mock-key"}',
            }
        }
    elif method == "POST" and url.endswith("/generations"):
        mock_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "mock-gen-job-67890"}
        }
    elif method == "GET" and "/generations/" in url:
        mock_resp.json.return_value = {
            "generations_by_pk": {
                "status": "COMPLETE",
                "generated_images": [
                    {"url": "https://mock-cdn.leonardo.ai/mock-render.jpg"}
                ],
            }
        }
    else:
        mock_resp.status_code = 404
        mock_resp.text = f"Mock route not found: {method} {url}"

    return mock_resp


def _mock_requests_post(url: str, **kwargs) -> MagicMock:
    """Intercepta `requests.post` usado no upload S3 da init-image."""
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_resp.text = ""
    return mock_resp


def _mock_requests_get(url: str, **kwargs) -> MagicMock:
    """Intercepta `requests.get` usado no download da imagem renderizada."""
    jpeg_bytes = _create_mock_gray_jpeg_bytes()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [jpeg_bytes]
    return mock_resp


def run_e2e_integration_test(image_path: str = "imgs/images.jpeg") -> Path:
    print("=" * 70)
    print("   RENDERX - TESTE DE INTEGRAÇÃO E2E (FASE 2 + 2.5 + 3 COM MOCK)")
    print("=" * 70)

    # Setup
    test_image = Path(image_path)
    if not test_image.is_file():
        print(f"[-] Erro: Imagem de teste não encontrada em '{test_image}'.")
        sys.exit(1)

    # [1/4] Visão (Fase 2)
    print(f"\n[1/4] Extraindo dados estruturados (FurnitureSpec) via Gemini OCR ({test_image})...")
    spec: FurnitureSpec = analyze_carpentry_image(
        image_input=test_image,
        additional_instructions="Acabamento em carvalho natural e laca fosca",
    )
    print(f"      -> Tipo de entrada : {spec.input_type}")
    print(f"      -> Tipo de móvel   : {spec.furniture_type}")
    print(
        f"      -> Dimensões (cm)  : {spec.overall_dimensions.width_cm} x "
        f"{spec.overall_dimensions.height_cm} x {spec.overall_dimensions.depth_cm} "
        f"(confiança: {spec.overall_dimensions.confidence})"
    )
    print(
        f"      -> Estrutura       : {spec.structure.doors_count} portas, "
        f"{spec.structure.drawers_count} gavetas, {spec.structure.shelves_count} prateleiras"
    )

    # [2/4] Estilos (Fase 2.5)
    print(f"\n[2/4] Carregando estilo YAML para '{spec.furniture_type}' (variante: modern_minimal)...")
    style_dict = load_style(spec.furniture_type, variant="modern_minimal")
    print(f"      -> Ambiente do estilo: {style_dict.get('environment', '')[:80]}...")

    # [3/4] Geração Programática de Prompts
    print("\n[3/4] Construindo prompts programaticamente (build_prompt & build_negative_prompt)...")
    sd_prompt = build_prompt(spec, style_dict)
    negative_prompt = build_negative_prompt(spec, style_dict)
    print(f"      -> sd_prompt      : {sd_prompt[:110]}...")
    print(f"      -> negative_prompt: {negative_prompt[:110]}...")

    # [4/4] Renderização (Fase 3 com Mock das chamadas HTTP do Leonardo.ai)
    print("\n[4/4] Executando render_module com Mock HTTP da API do Leonardo.ai...")
    leonardo_key = os.getenv("LEONARDO_API_KEY") or "mock-leonardo-api-key"

    with (
        patch("render_module.requests.request", side_effect=_mock_requests_request),
        patch("render_module.requests.post", side_effect=_mock_requests_post),
        patch("render_module.requests.get", side_effect=_mock_requests_get),
    ):
        rendered_image_path = render_carpentry_design(
            image_path=test_image,
            sd_prompt=sd_prompt,
            negative_prompt=negative_prompt,
            spec_data=spec,
            input_type=spec.input_type,
            strength_type="High",
            api_key=leonardo_key,
            outputs_base_dir="outputs",
        )

    job_dir = rendered_image_path.parent
    print("\n" + "=" * 70)
    print("[+] TESTE DE INTEGRAÇÃO E2E CONCLUÍDO COM SUCESSO!")
    print(f"    Pasta do Job gerada : {job_dir.resolve()}")
    print("    Arquivos salvos na pasta:")
    for item in sorted(job_dir.iterdir()):
        print(f"      - {item.name} ({item.stat().st_size} bytes)")
    print("=" * 70)

    return job_dir


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "imgs/images.jpeg"
    run_e2e_integration_test(img_arg)
