"""
Módulo de Visão e OCR com Google Gemini - RenderX (Fase 2)
Responsável por analisar imagens de marcenaria (prints 3D ou rascunhos em papel),
extrair especificações geométricas e técnicas via Pydantic e construir
prompts fotorrealistas de forma programática.
"""

import os
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Literal
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

# Carrega variáveis de ambiente
load_dotenv()


# ============================================================================
# Schemas Estruturados (Pydantic)
# ============================================================================

class Dimensions(BaseModel):
    """Dimensões geométricas estimadas ou lidas via cotas técnicas."""
    width_cm: Optional[float] = Field(
        default=None,
        description="Largura total em centímetros extraída de cotas ou estimada."
    )
    height_cm: Optional[float] = Field(
        default=None,
        description="Altura total em centímetros extraída de cotas ou estimada."
    )
    depth_cm: Optional[float] = Field(
        default=None,
        description="Profundidade total em centímetros extraída de cotas ou estimada."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grau de certeza/confiança na leitura das dimensões (0.0 a 1.0)."
    )


class Structure(BaseModel):
    """Contagem e detalhes estruturais dos componentes de marcenaria."""
    doors_count: int = Field(
        default=0,
        ge=0,
        description="Número exato de portas visíveis ou anotadas."
    )
    drawers_count: int = Field(
        default=0,
        ge=0,
        description="Número exato de gavetas visíveis ou anotadas."
    )
    shelves_count: int = Field(
        default=0,
        ge=0,
        description="Número de prateleiras internas ou externas visíveis ou anotadas."
    )
    niches_count: int = Field(
        default=0,
        ge=0,
        description="Número de nichos ou vãos abertos."
    )
    details: Optional[str] = Field(
        default="",
        description="Detalhes estruturais específicos: altura dos pés, espessuras, divisórias internas, tipo de puxador."
    )


class FurnitureSpec(BaseModel):
    """Especificação técnica completa extraída da imagem do móvel."""
    input_type: Literal["DIGITAL_3D", "PAPER_SKETCH"] = Field(
        ...,
        description="Classificação da imagem: DIGITAL_3D (print de software CAD/3D) ou PAPER_SKETCH (esboço manual em papel)."
    )
    furniture_type: str = Field(
        ...,
        description="Tipo ou categoria do móvel identificado (ex: aparador, armário de cozinha, guarda-roupa, estante, cômoda, mesa)."
    )
    overall_dimensions: Dimensions = Field(
        ...,
        description="Dimensões gerais do móvel e nível de confiança do OCR."
    )
    structure: Structure = Field(
        ...,
        description="Contagem precisa e particularidades da estrutura do móvel."
    )
    materials: List[str] = Field(
        default_factory=list,
        description="Lista de materiais, acabamentos e texturas detectados ou especificados (ex: carvalho natural, laca fosca, metal preto)."
    )
    environment: Optional[str] = Field(
        default="",
        description="Ambiente residencial ou comercial sugerido para o móvel (ex: sala de estar, cozinha, dormitório)."
    )
    handwritten_notes_raw: List[str] = Field(
        default_factory=list,
        description="Texto bruto, números e anotações técnicas lidos na imagem via OCR."
    )
    uncertain_fields: List[str] = Field(
        default_factory=list,
        description="Lista de campos, cotas ou detalhes ambíguos ou ilegíveis que demandam confirmação humana."
    )
    negative_constraints: List[str] = Field(
        default_factory=list,
        description="Restrições negativas para manter fidelidade geométrica (ex: sem gavetas adicionais, sem puxadores externos, sem nichos abertos)."
    )


# ============================================================================
# Instruções do Sistema para OCR e Análise Técnica
# ============================================================================

SYSTEM_INSTRUCTION = """
Você é um Arquiteto Sênior especialista em Engenharia de Marcenaria e Design de Mobiliário, atuando com precisão milimétrica de OCR técnico.
Sua missão é inspecionar minuciosamente a imagem fornecida (seja ela um print volumétrico de software 3D ou um esboço manual/croqui em papel com cotas) e extrair os dados técnicos estruturados de acordo com o schema fornecido.

Diretrizes obrigatórias:
1. INPUT_TYPE:
   - "DIGITAL_3D": Imagens renderizadas, capturas de softwares 3D (SketchUp, Promob, AutoCAD, Revit), modelos poligonais com superfícies ou wireframes.
   - "PAPER_SKETCH": Desenhos feitos à mão livre sobre papel, croquis manuais, anotações de cota a caneta/lápis, linhas de rascunho.
2. DIMENSÕES E COTAS:
   - Leia atentamente todas as cotas numéricas (largura, altura, profundidade, altura de pés, vãos).
   - Indique em `confidence` um valor de 0.0 a 1.0 com base na clareza visual dos números.
   - Se uma medida estiver borrada ou ambígua, liste a dúvida em `uncertain_fields`.
3. ESTRUTURA E CONTAGEM EXATA:
   - Conte com exatidão o número de portas, gavetas, nichos e prateleiras. NÃO invente divisões não presentes.
   - Descreva detalhes mecânicos em `structure.details` (ex: "pés de 30cm, prateleiras reguláveis com furação de 6.4cm").
4. MATERIAIS E ACABAMENTOS:
   - Identifique padrões visíveis e incorpore qualquer instrução complementar informada pelo marceneiro.
5. RESTRIÇÕES NEGATIVAS (negative_constraints):
   - Enumere elementos que NÃO devem ser gerados no render para não alterar a marcenaria (ex: se o móvel tem apenas portas e nenhuma gaveta, restrinja explicitamente "gavetas extras", "nichos abertos").
"""


# ============================================================================
# Função de Geração Programática de Prompts
# ============================================================================

def build_prompt(spec: FurnitureSpec, style: Optional[Dict[str, Any]] = None) -> str:
    """
    Constrói programaticamente o prompt fotorrealista (sd_prompt) a partir
    dos campos estruturados de FurnitureSpec e de configurações de estilo.

    O Gemini NÃO escreve mais o prompt final; a fidelidade estrutural é mantida
    concatenando de forma determinística contagens, materiais e dimensões.

    Args:
        spec: Instância de FurnitureSpec extraída via OCR.
        style: Dicionário opcional com variantes de estilo contendo:
               - camera (ex: 'eye-level architectural shot')
               - lighting (ex: 'soft diffused daylight')
               - environment (ex: 'minimalist Scandinavian living room')
               - quality_tags (ex: 'photorealistic, 8k resolution, ray tracing')

    Returns:
        String com o prompt detalhado para Leonardo.ai / SDXL.
    """
    style = style or {}
    parts: List[str] = []

    # 1. Mapeamento e Assunto Principal
    pt_to_en = {
        "aparador": "sideboard buffet credenza",
        "armario": "cabinet",
        "armário": "cabinet",
        "armário de cozinha": "kitchen cabinetry",
        "guarda-roupa": "wardrobe closet",
        "estante": "bookshelf wall unit",
        "comoda": "dresser chest of drawers",
        "cômoda": "dresser chest of drawers",
        "mesa": "table",
        "rack": "TV stand media console",
        "balcao": "counter cabinet",
        "balcão": "counter cabinet",
    }
    raw_type = spec.furniture_type.strip().lower()
    subject_type = pt_to_en.get(raw_type, spec.furniture_type.strip())
    parts.append(f"Professional architectural photography of a bespoke high-end {subject_type}")

    # 2. Fidelidade Estrutural (Contagens Exatas)
    struct = spec.structure
    counts: List[str] = []
    if struct.doors_count > 0:
        counts.append(f"exactly {struct.doors_count} hinged door{'s' if struct.doors_count > 1 else ''}")
    if struct.drawers_count > 0:
        counts.append(f"exactly {struct.drawers_count} drawer{'s' if struct.drawers_count > 1 else ''}")
    if struct.shelves_count > 0:
        counts.append(f"{struct.shelves_count} internal {'shelf' if struct.shelves_count == 1 else 'shelves'}")
    if struct.niches_count > 0:
        counts.append(f"{struct.niches_count} open {'niche' if struct.niches_count == 1 else 'niches'}")

    if counts:
        parts.append("featuring " + ", ".join(counts))

    if struct.details:
        parts.append(struct.details.strip())

    # 3. Dimensões e Proporções
    dims = spec.overall_dimensions
    dim_tokens: List[str] = []
    if dims.width_cm is not None:
        dim_tokens.append(f"{dims.width_cm:g}cm width")
    if dims.height_cm is not None:
        dim_tokens.append(f"{dims.height_cm:g}cm height")
    if dims.depth_cm is not None:
        dim_tokens.append(f"{dims.depth_cm:g}cm depth")

    if dim_tokens:
        parts.append("exact proportions of " + " x ".join(dim_tokens))

    # 4. Materiais e Acabamentos
    if spec.materials:
        materials_str = ", ".join(spec.materials)
        parts.append(f"crafted from {materials_str}")

    # 5. Ambiente (Prioriza o estilo se definido, senão usa o spec)
    env = style.get("environment") or spec.environment or "modern luxury interior space"
    parts.append(f"situated in a {env}")

    # 6. Câmera e Enquadramento (Injetado pelo Estilo)
    camera = style.get(
        "camera",
        "straight-on eye-level architectural shot, perfectly centered composition, clean geometric perspective"
    )
    if camera:
        parts.append(camera)

    # 7. Iluminação (Injetado pelo Estilo)
    lighting = style.get(
        "lighting",
        "soft studio lighting, warm natural ambient daylight fill, realistic soft contact shadows"
    )
    if lighting:
        parts.append(lighting)

    # 8. Tags de Qualidade e Render
    quality = style.get(
        "quality_tags",
        "photorealistic, 8k resolution, architectural digest photography, hyper-detailed wood grain, sharp focus, ray tracing"
    )
    if quality:
        parts.append(quality)

    return ", ".join(p.rstrip(".,") for p in parts if p)


def build_negative_prompt(spec: FurnitureSpec, style: Optional[Dict[str, Any]] = None) -> str:
    """
    Constrói programaticamente o prompt negativo combinando tags de qualidade
    com as restrições negativas extraídas na análise da marcenaria.
    """
    style = style or {}
    base_negatives = [
        "distorted geometry",
        "blurry",
        "low quality",
        "cartoon",
        "cgi render artifacts",
        "extra legs",
        "deformed hardware",
        "unaligned doors",
    ]

    # Adiciona restrições do spec (ex: "sem gavetas", "sem nichos")
    for constraint in spec.negative_constraints:
        clean = constraint.strip().lower()
        if clean.startswith("sem "):
            clean = clean[4:].strip()
        elif clean.startswith("no "):
            clean = clean[3:].strip()
        elif clean.startswith("não possui "):
            clean = clean[11:].strip()
        elif clean.startswith("nao possui "):
            clean = clean[11:].strip()
        if clean:
            base_negatives.append(f"no {clean}")

    # Adiciona tags negativas customizadas do estilo
    style_negatives = style.get("negative") or style.get("negative_tags")
    if style_negatives:
        if isinstance(style_negatives, list):
            base_negatives.extend(style_negatives)
        elif isinstance(style_negatives, str):
            base_negatives.append(style_negatives)

    return ", ".join(base_negatives)


# ============================================================================
# Função de Inferência com Google Gemini
# ============================================================================

def analyze_carpentry_image(
    image_input: Union[str, Path, Image.Image],
    additional_instructions: str = "",
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.1-flash-lite",
    max_retries: int = 3
) -> FurnitureSpec:
    """
    Analisa uma imagem de projeto de marcenaria usando o Google Gemini oficial (google-genai)
    forçando saída estruturada através do schema Pydantic `FurnitureSpec`.

    Args:
        image_input: Caminho da imagem (str/Path) ou instância de PIL.Image.Image.
        additional_instructions: Notas ou requisitos adicionais do marceneiro/designer.
        api_key: Chave de API do Gemini (busca GEMINI_API_KEY no ambiente se omitido).
        model_name: Modelo do Gemini (padrão: gemini-3.1-flash-lite).
        max_retries: Número de tentativas em caso de erro temporário 503/429.

    Returns:
        Instância validada de `FurnitureSpec`.

    Raises:
        ValueError: Se a credencial for inválida ou a imagem não for encontrada.
        RuntimeError: Se a API falhar após as tentativas ou o schema for violado.
    """
    # 1. Validação de credenciais
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "sua_chave_gemini_aqui":
        raise ValueError(
            "GEMINI_API_KEY não configurada. Defina a variável de ambiente ou passe api_key."
        )

    # 2. Carregamento da imagem via Pillow
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        image = Image.open(image_path)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("O parâmetro image_input deve ser um caminho de arquivo ou uma instância de PIL.Image.")

    # 3. Inicialização do Cliente oficial google-genai
    client = genai.Client(api_key=gemini_key)

    # Configuração com temperatura 0.2 para foco estrito em OCR e precisão
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=FurnitureSpec,
        temperature=0.2
    )

    # 4. Construção dos conteúdos
    prompt_text = (
        "Analise minuciosamente esta imagem técnica de marcenaria. "
        "Extraia as dimensões, a contagem estrutural de portas/gavetas/prateleiras/nichos, "
        "o tipo de imagem (DIGITAL_3D ou PAPER_SKETCH) e todas as anotações visíveis conforme o schema."
    )
    if additional_instructions.strip():
        prompt_text += f"\nInstruções adicionais do marceneiro/designer: {additional_instructions.strip()}"

    # 5. Execução da inferência com retry para resiliência contra picos de demanda
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt_text],
                config=config
            )

            # Validação e retorno do schema tipado
            if response.parsed and isinstance(response.parsed, FurnitureSpec):
                return response.parsed

            if response.text:
                return FurnitureSpec.model_validate_json(response.text)

            raise RuntimeError("A API do Gemini retornou uma resposta vazia.")

        except (ServerError, APIError) as e:
            last_error = e
            # Se for erro temporário de alta demanda (503 ou 429), tenta backoff exponencial
            status_code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status_code in (503, 429) or "UNAVAILABLE" in str(e):
                wait_time = attempt * 2
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Erro na comunicação com a API do Gemini: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Erro inesperado durante a análise de visão: {e}") from e

    raise RuntimeError(
        f"Falha ao consultar Gemini após {max_retries} tentativas devido a sobrecarga no serviço: {last_error}"
    ) from last_error


# ============================================================================
# Execução CLI para Testes e Validação
# ============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Uso para teste manual:")
        print("  python vision_module.py <caminho_da_imagem> [instrucoes_adicionais]")
        sys.exit(1)

    test_img = sys.argv[1]
    extra_notes = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"[*] Analisando imagem técnica: {test_img}")
    if extra_notes:
        print(f"[*] Instruções adicionais: {extra_notes}")

    try:
        spec_result = analyze_carpentry_image(test_img, additional_instructions=extra_notes)

        print("\n" + "=" * 60)
        print("[+] 1. Especificação Estruturada Extraída (FurnitureSpec):")
        print("=" * 60)
        print(spec_result.model_dump_json(indent=2))

        print("\n" + "=" * 60)
        print("[+] 2. Prompt Fotorrealista Gerado Programaticamente (build_prompt):")
        print("=" * 60)
        generated_sd_prompt = build_prompt(spec_result)
        print(generated_sd_prompt)

        print("\n" + "=" * 60)
        print("[+] 3. Prompt Negativo Gerado Programaticamente:")
        print("=" * 60)
        generated_negative_prompt = build_negative_prompt(spec_result)
        print(generated_negative_prompt)

    except Exception as err:
        print(f"\n[-] Erro durante análise: {err}")
        sys.exit(1)
