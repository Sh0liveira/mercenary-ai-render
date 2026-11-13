"""
Módulo Juiz Gemini (QA Automático) - RenderX (Fase 3.5)
Responsável por atuar como Inspetor de Qualidade rigoroso de marcenaria,
comparando a imagem gerada pelo motor de renderização com as especificações
técnicas originais (`spec_data`) e as diretrizes de estilo (`style_data`).
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

# Carrega variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"


# ============================================================================
# Schema Estruturado de Avaliação (Pydantic)
# ============================================================================

class JudgeEvaluation(BaseModel):
    """Resultado estruturado da inspeção de qualidade (QA) da imagem renderizada."""
    is_approved: bool = Field(
        ...,
        description="True se a imagem gerada for fiel e aceitável; False se houver erros estruturais graves ou alucinações severas."
    )
    adherence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Nota de 0 a 100 indicando a fidelidade geral ao projeto original e ao estilo solicitado."
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Lista de elementos previstos no spec_data que o renderizador esqueceu de desenhar (ex: 'faltam 2 gavetas')."
    )
    hallucinations: List[str] = Field(
        default_factory=list,
        description="Lista de artefatos bizarros ou elementos extras inventados (ex: 'puxador flutuando', '3 pés em vez de 4')."
    )
    reasoning: str = Field(
        ...,
        description="Justificativa objetiva e breve explicando a aprovação ou reprovação da imagem."
    )


# ============================================================================
# Instruções do Sistema para o Inspetor de Qualidade
# ============================================================================

JUDGE_SYSTEM_INSTRUCTION = """
Você é um Inspetor de Qualidade Sênior de Marcenaria e Visualização Arquitetônica (QA).
Sua função é comparar rigorosamente a imagem renderizada fornecida com as especificações técnicas (`spec_data`) e o estilo esperado (`style_data`).

Critérios de Avaliação Rígidos:
1. FIDELIDADE ESTRUTURAL (Prioridade Máxima):
   - Verifique se a contagem de portas, gavetas, prateleiras e nichos na imagem bate com `spec_data.structure`.
   - Verifique se as proporções gerais (largura x altura x profundidade) são coerentes com `spec_data.overall_dimensions`.
   - Verifique se há restrições negativas violadas (`spec_data.negative_constraints`).
2. DETECÇÃO DE ALUCINAÇÕES E ARTEFATOS:
   - Procure defeitos de geração de IA: puxadores flutuando ou duplicados, pés assimétricos/faltando, portas tortas ou desalinhadas, texturas derretidas.
3. ADERÊNCIA AO ESTILO E MATERIAIS:
   - Avalie se os materiais (`spec_data.materials`) e o ambiente/iluminação (`style_data`) foram respeitados.
4. REGRA DE APROVAÇÃO (`is_approved` e `adherence_score`):
   - Erros graves de contagem estrutural (ex: portas ou gavetas a mais/a menos) ou alucinações severas DEVEM reprovar (`is_approved = False`) e reduzir drasticamente o `adherence_score`.
   - Seja objetivo, técnico e conciso no campo `reasoning`.
"""


# ============================================================================
# Função Principal de Avaliação
# ============================================================================

def evaluate_render(
    generated_image_path: Union[str, Path],
    spec_data: Union[Dict[str, Any], BaseModel],
    style_data: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 3,
) -> JudgeEvaluation:
    """
    Avalia a qualidade e fidelidade estrutural de uma imagem renderizada comparando-a
    com as especificações do projeto (`spec_data`) e estilo (`style_data`).

    Args:
        generated_image_path: Caminho da imagem gerada pelo motor de renderização.
        spec_data: Dicionário (ou modelo Pydantic `FurnitureSpec`) com o projeto original.
        style_data: Dicionário com as regras de estilo aplicadas (luz, ambiente, câmera).
        api_key: Chave da API Gemini (opcional; usa GEMINI_API_KEY do ambiente se omitido).
        model_name: Modelo Gemini a ser usado (padrão do projeto: gemini-3.1-flash-lite).
        max_retries: Número máximo de tentativas em erros transitórios (503/429).

    Returns:
        Instância validada de `JudgeEvaluation`.

    Raises:
        ValueError: Se a chave de API não estiver configurada.
        FileNotFoundError: Se a imagem gerada não existir no caminho informado.
        RuntimeError: Se a API falhar após as tentativas ou retornar resposta inválida.
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "sua_chave_gemini_aqui":
        raise ValueError(
            "GEMINI_API_KEY não configurada. Defina a variável de ambiente ou passe api_key."
        )

    img_path = Path(generated_image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Imagem gerada não encontrada para avaliação: {img_path}")

    image = Image.open(img_path)

    # Normaliza spec_data e style_data para serialização JSON legível
    spec_dict = spec_data.model_dump() if isinstance(spec_data, BaseModel) else (spec_data or {})
    style_dict = style_data or {}

    prompt_text = (
        "Realize uma inspeção rigorosa de QA na imagem renderizada anexa.\n"
        "Compare-a com a especificação técnica original e o estilo abaixo:\n\n"
        f"=== SPEC_DATA (Projeto Original) ===\n{json.dumps(spec_dict, indent=2, ensure_ascii=False)}\n\n"
        f"=== STYLE_DATA (Estilo Esperado) ===\n{json.dumps(style_dict, indent=2, ensure_ascii=False)}\n\n"
        "Retorne a avaliação estritamente conforme o schema `JudgeEvaluation`."
    )

    client = genai.Client(api_key=gemini_key)
    config = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=JudgeEvaluation,
        temperature=0.1,
    )

    logger.info("Iniciando avaliação QA do render '%s' com modelo '%s'...", img_path.name, model_name)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt_text],
                config=config,
            )

            if response.parsed and isinstance(response.parsed, JudgeEvaluation):
                evaluation = response.parsed
            elif response.text:
                evaluation = JudgeEvaluation.model_validate_json(response.text)
            else:
                raise RuntimeError("A API do Gemini retornou uma resposta vazia na avaliação QA.")

            logger.info(
                "Avaliação QA concluída | Aprovado: %s | Score: %d/100",
                evaluation.is_approved,
                evaluation.adherence_score,
            )
            return evaluation

        except (ServerError, APIError) as e:
            last_error = e
            status_code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status_code in (503, 429) or "UNAVAILABLE" in str(e):
                wait_time = attempt * 2
                logger.warning(
                    "Erro transitório na API Gemini durante avaliação QA (tentativa %d/%d): %s. Aguardando %ds...",
                    attempt,
                    max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
                continue

            logger.error("Erro permanente na API Gemini durante avaliação QA: %s", e, exc_info=True)
            raise RuntimeError(f"Falha na comunicação com a API do Gemini (QA): {e}") from e

        except Exception as e:
            logger.error("Erro inesperado ao avaliar render '%s': %s", img_path, e, exc_info=True)
            raise RuntimeError(f"Erro inesperado durante a avaliação QA: {e}") from e

    logger.error(
        "Falha definitiva na avaliação QA após %d tentativas: %s",
        max_retries,
        last_error,
    )
    raise RuntimeError(
        f"Falha ao consultar Gemini QA após {max_retries} tentativas: {last_error}"
    ) from last_error


# ============================================================================
# Execução CLI para Teste Manual
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso para teste manual:")
        print("  python judge_module.py <caminho_imagem_gerada> <caminho_spec_json> [caminho_style_yaml_ou_json]")
        sys.exit(1)

    gen_img_arg = sys.argv[1]
    spec_json_arg = Path(sys.argv[2])

    if not spec_json_arg.is_file():
        print(f"[-] Arquivo spec.json não encontrado: {spec_json_arg}")
        sys.exit(1)

    loaded_spec = json.loads(spec_json_arg.read_text(encoding="utf-8"))
    loaded_style: Dict[str, Any] = {}

    if len(sys.argv) > 3:
        style_arg = Path(sys.argv[3])
        if style_arg.is_file():
            loaded_style = json.loads(style_arg.read_text(encoding="utf-8"))

    try:
        result = evaluate_render(
            generated_image_path=gen_img_arg,
            spec_data=loaded_spec,
            style_data=loaded_style,
        )
        print("\n[+] Resultado da Avaliação QA (JudgeEvaluation):")
        print(result.model_dump_json(indent=2))
    except Exception as err:
        print(f"\n[-] Erro durante avaliação QA: {err}")
        sys.exit(1)
