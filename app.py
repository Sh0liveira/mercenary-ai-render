"""
WoodAI Studio - Interface Web Principal (Fase 4)
Orquestra o pipeline completo do RenderX em Streamlit:
OCR Estruturado (Gemini) -> Biblioteca de Estilos (YAML) -> Renderização (Leonardo.ai) -> Juiz QA (Gemini).
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

import streamlit as st
from PIL import Image

from vision_module import (
    FurnitureSpec,
    analyze_carpentry_image,
    build_prompt,
    build_negative_prompt,
)
from style_loader import load_style, list_available_styles
from render_module import render_carpentry_design
from judge_module import evaluate_render, JudgeEvaluation


# ============================================================================
# Configuração da Página (tema em .streamlit/config.toml)
# ============================================================================

st.set_page_config(
    page_title="WoodAI Studio",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tema base (cores de fundo, sidebar, texto e destaque) definido em .streamlit/config.toml.
# CSS abaixo é apenas cirúrgico: header nativo, inputs e cards de veredito.
CUSTOM_CSS = """
<style>
    /* Header transparente: some a barra preta, mas mantém o botão de reabrir a sidebar */
    header[data-testid="stHeader"] {background: transparent;}
    /* NÃO esconder stToolbar: o botão de reabrir a sidebar vive dentro dele */
    [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
    [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
    #MainMenu, footer {display: none !important;}

    /* Sidebar fixa: remove o botão de recolher para ela nunca sumir */
    [data-testid="stSidebarCollapseButton"] {display: none !important;}

    /* Botão para reabrir a sidebar (nomes variam entre versões do Streamlit) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
    }
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background-color: #D97736 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        color: #FFFFFF !important;
    }
    .block-container {padding-top: 2rem; max-width: 1280px;}

    /* Texto dentro dos inputs e placeholders */
    input, textarea,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #3B2715 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #3B2715 !important;
        opacity: 0.55;
    }

    /* Cards de veredito do Juiz */
    .verdict-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
        border: 1px solid transparent;
    }
    .verdict-approved {
        background-color: #EEF7EF;
        border-color: #CFE6D1;
        color: #2D4A30;
    }
    .verdict-rejected {
        background-color: #FFF4E8;
        border-color: #F3D4B5;
        color: #3B2715;
    }
    .verdict-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .verdict-card ul {margin: 4px 0 12px 0; padding-left: 20px;}
    .header-subtitle {
        color: #3B2715;
        opacity: 0.7;
        font-size: 1.05rem;
        margin-top: -8px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# Helpers de Mock para Renderização (quando sem saldo na API Leonardo.ai)
# ============================================================================

def _create_mock_gray_jpeg_bytes(width: int = 1024, height: int = 768) -> bytes:
    """Gera uma imagem cinza simples via Pillow para simular o retorno do Leonardo.ai."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _mock_requests_request(method: str, url: str, **kwargs) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    if method == "POST" and url.endswith("/init-image"):
        mock_resp.json.return_value = {
            "uploadInitImage": {
                "id": "mock-init-img-ui",
                "url": "https://mock-s3.local/upload",
                "fields": '{"key": "mock-key"}',
            }
        }
    elif method == "POST" and url.endswith("/generations"):
        mock_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "mock-gen-job-ui"}
        }
    elif method == "GET" and "/generations/" in url:
        mock_resp.json.return_value = {
            "generations_by_pk": {
                "status": "COMPLETE",
                "generated_images": [{"url": "https://mock-cdn.leonardo.ai/render.jpg"}],
            }
        }
    else:
        mock_resp.status_code = 404
    return mock_resp


def _mock_requests_post(url: str, **kwargs) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    return mock_resp


def _mock_requests_get(url: str, **kwargs) -> MagicMock:
    jpeg_bytes = _create_mock_gray_jpeg_bytes()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [jpeg_bytes]
    return mock_resp


def _execute_render(
    image_path: Path,
    sd_prompt: str,
    negative_prompt: str,
    spec: FurnitureSpec,
    leonardo_key: str,
    use_mock: bool,
) -> Path:
    """Executa o render_carpentry_design com ou sem mock HTTP."""
    render_kwargs = {
        "image_path": image_path,
        "sd_prompt": sd_prompt,
        "negative_prompt": negative_prompt,
        "spec_data": spec,
        "input_type": spec.input_type,
        "strength_type": "High",
        "api_key": leonardo_key,
        "outputs_base_dir": "outputs",
    }
    if use_mock:
        with (
            patch("render_module.requests.request", side_effect=_mock_requests_request),
            patch("render_module.requests.post", side_effect=_mock_requests_post),
            patch("render_module.requests.get", side_effect=_mock_requests_get),
        ):
            return render_carpentry_design(**render_kwargs)
    return render_carpentry_design(**render_kwargs)


# ============================================================================
# Cabeçalho Principal
# ============================================================================

st.title("🪵 WoodAI Studio")
st.markdown(
    '<p class="header-subtitle">'
    "Renderização fotorrealista para projetos de marcenaria com fidelidade geométrica "
    "(OCR Estruturado • Estilos YAML • ControlNet SDXL • Auditoria QA)"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()


# ============================================================================
# Barra Lateral (Sidebar) - Controles do Projeto
# ============================================================================

with st.sidebar:
    st.header("Configuração do Projeto")

    uploaded_file = st.file_uploader(
        "Envie o rascunho ou print 3D",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceitos: JPG, JPEG ou PNG.",
    )

    # Descobre variantes disponíveis nos YAMLs ou usa padrão
    catalog = list_available_styles()
    all_variants: List[str] = sorted(
        {v for variants in catalog.values() for v in variants}
    ) or ["modern_minimal", "classic_warm"]

    default_idx = all_variants.index("modern_minimal") if "modern_minimal" in all_variants else 0
    selected_style = st.selectbox(
        "Estilo Visual",
        options=all_variants,
        index=default_idx,
        help="Define câmera, ambiente e iluminação sem alterar a estrutura do móvel.",
    )

    additional_notes = st.text_area(
        "Materiais e Instruções Adicionais (Opcional)",
        placeholder="Ex: Acabamento em carvalho natural e laca fosca, puxadores ocultos...",
        height=90,
    )

    use_mock_render = st.checkbox(
        "Modo Simulação (Mock Leonardo.ai)",
        value=True,
        help="Simula a chamada web do Leonardo.ai localmente sem consumir créditos de API.",
    )

    st.divider()
    generate_clicked = st.button("Gerar Projeto", type="primary")


# ============================================================================
# Orquestração do Pipeline
# ============================================================================

if generate_clicked:
    if not uploaded_file:
        st.warning("Por favor, envie uma imagem do projeto na barra lateral antes de continuar.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
        temp_image_path = Path(tmp_dir) / f"input_upload{suffix}"
        temp_image_path.write_bytes(uploaded_file.getbuffer())

        try:
            with st.status("Executando pipeline WoodAI Studio...", expanded=True) as status_box:
                # Passo 1: Visão e OCR (Gemini Pydantic)
                st.write("🔍 **Passo 1/3:** Analisando geometria e cotas via Gemini OCR...")
                spec: FurnitureSpec = analyze_carpentry_image(
                    image_input=temp_image_path,
                    additional_instructions=additional_notes,
                )

                # Carregamento de Estilo e Geração Inicial de Prompts
                style_dict: Dict[str, Any] = load_style(spec.furniture_type, variant=selected_style)
                sd_prompt = build_prompt(spec, style_dict)
                negative_prompt = build_negative_prompt(spec, style_dict)
                leonardo_key = os.getenv("LEONARDO_API_KEY") or "mock-leonardo-api-key"

                # Loop de Retentativa Inteligente (Agente de Auto-Reflexão)
                max_attempts = 3
                attempt = 1
                rendered_path: Optional[Path] = None
                evaluation: Optional[JudgeEvaluation] = None
                attempt_history: List[Dict[str, Any]] = []

                while attempt <= max_attempts:
                    status_box.update(
                        label=f"Executando pipeline WoodAI Studio — Tentativa {attempt}/{max_attempts}...",
                        state="running",
                    )

                    # Passo 2: Renderização (Leonardo.ai ou Mock)
                    st.write(
                        f"🎨 **Passo 2/3 (Tentativa {attempt}/{max_attempts}):** "
                        "Renderizando projeto com ControlNet SDXL..."
                    )
                    rendered_path = _execute_render(
                        image_path=temp_image_path,
                        sd_prompt=sd_prompt,
                        negative_prompt=negative_prompt,
                        spec=spec,
                        leonardo_key=leonardo_key,
                        use_mock=use_mock_render,
                    )

                    # Passo 3: Juiz QA (Gemini)
                    st.write(
                        f"⚖️ **Passo 3/3 (Tentativa {attempt}/{max_attempts}):** "
                        "Auditando fidelidade estrutural com o Juiz Gemini..."
                    )
                    evaluation = evaluate_render(
                        generated_image_path=rendered_path,
                        spec_data=spec.model_dump(),
                        style_data=style_dict,
                    )

                    attempt_history.append({
                        "attempt": attempt,
                        "is_approved": evaluation.is_approved,
                        "adherence_score": evaluation.adherence_score,
                        "sd_prompt": sd_prompt,
                        "negative_prompt": negative_prompt,
                        "missing_elements": evaluation.missing_elements,
                        "hallucinations": evaluation.hallucinations,
                    })

                    if evaluation.is_approved:
                        break

                    # Retroalimentação de prompts caso haja próxima tentativa
                    if attempt < max_attempts:
                        st.warning(
                            f"⚠️ Tentativa {attempt}/{max_attempts}: Reprovado pelo QA. "
                            "Ajustando prompts e tentando novamente..."
                        )
                        if evaluation.missing_elements:
                            missing_items = ", ".join(evaluation.missing_elements)
                            sd_prompt = f"{sd_prompt}, MUST INCLUDE: {missing_items}"
                        if evaluation.hallucinations:
                            hallucination_items = ", ".join(evaluation.hallucinations)
                            negative_prompt = f"{negative_prompt}, AVOID: {hallucination_items}"

                    attempt += 1

                if evaluation and evaluation.is_approved:
                    status_box.update(
                        label=f"Pipeline concluído e aprovado na Tentativa {attempt}/{max_attempts}!",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label=f"Pipeline finalizado após {max_attempts} tentativas (reprovado pelo QA).",
                        state="error",
                        expanded=False,
                    )

            assert rendered_path is not None and evaluation is not None

            if not evaluation.is_approved:
                st.error(
                    f"❌ O projeto não foi aprovado pelo Juiz QA após {max_attempts} tentativas "
                    f"(Score final: {evaluation.adherence_score}/100). Veja os detalhes da última auditoria abaixo."
                )

            # ----------------------------------------------------------------
            # Exibição de Especificações Extraídas
            # ----------------------------------------------------------------
            st.subheader("📐 Especificação Técnica Extraída")
            m1, m2, m3, m4 = st.columns(4, gap="medium")
            dims = spec.overall_dimensions
            dim_text = (
                f"{dims.width_cm or '?'} x {dims.height_cm or '?'} x {dims.depth_cm or '?'} cm"
            )
            m1.metric("Tipo de Móvel", spec.furniture_type)
            m2.metric("Dimensões (L x A x P)", dim_text)
            m3.metric("Portas / Gavetas", f"{spec.structure.doors_count}P / {spec.structure.drawers_count}G")
            m4.metric("Confiança OCR", f"{int(dims.confidence * 100)}%")

            with st.expander("Ver JSON completo extraído (FurnitureSpec), Prompts e Histórico de Tentativas"):
                st.json(spec.model_dump())
                st.markdown("**`sd_prompt` final:**")
                st.code(sd_prompt, language="text")
                st.markdown("**`negative_prompt` final:**")
                st.code(negative_prompt, language="text")
                st.markdown("**Histórico de Auto-Reflexão (Tentativas):**")
                st.json(attempt_history)

            st.divider()

            # ----------------------------------------------------------------
            # Comparativo Visual Lado a Lado
            # ----------------------------------------------------------------
            st.subheader("🖼️ Comparativo Visual")
            col_orig, col_rend = st.columns(2, gap="large")
            with col_orig:
                st.markdown("**Imagem Original**")
                st.image(Image.open(temp_image_path), use_container_width=True)
            with col_rend:
                st.markdown("**Imagem Renderizada**")
                st.image(Image.open(rendered_path), use_container_width=True)

            st.caption(f"Artefatos do job salvos em: `{rendered_path.parent}`")

            st.divider()

            # ----------------------------------------------------------------
            # Veredito do Juiz
            # ----------------------------------------------------------------
            st.subheader("⚖️ Veredito do Juiz (QA Automático)")
            if evaluation.is_approved:
                verdict_class = "verdict-approved"
                verdict_badge = f"✅ APROVADO — Score de Aderência: {evaluation.adherence_score}/100"
            else:
                verdict_class = "verdict-rejected"
                verdict_badge = f"⚠️ REPROVADO NA AUDITORIA — Score de Aderência: {evaluation.adherence_score}/100"

            missing_html = (
                "<ul>" + "".join(f"<li>{item}</li>" for item in evaluation.missing_elements) + "</ul>"
                if evaluation.missing_elements
                else "<p>Nenhum elemento faltante identificado.</p>"
            )
            hallucinations_html = (
                "<ul>" + "".join(f"<li>{item}</li>" for item in evaluation.hallucinations) + "</ul>"
                if evaluation.hallucinations
                else "<p>Nenhuma alucinação identificada.</p>"
            )

            verdict_html = (
                f'<div class="verdict-card {verdict_class}">'
                f'<div class="verdict-title">{verdict_badge}</div>'
                f"<p><strong>Parecer Técnico:</strong> {evaluation.reasoning}</p>"
                f"<p><strong>Elementos Faltantes:</strong></p>{missing_html}"
                f"<p><strong>Alucinações / Artefatos:</strong></p>{hallucinations_html}"
                "</div>"
            )
            st.markdown(verdict_html, unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Erro durante a execução do pipeline: {exc}")

elif uploaded_file is not None:
    st.info("Imagem carregada! Selecione o estilo na barra lateral e clique em **Gerar Projeto**.")
else:
    st.info("👈 Faça o upload de uma imagem de marcenaria na barra lateral para iniciar.")
    