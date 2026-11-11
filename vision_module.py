"""
Módulo de Visão e OCR com Google Gemini - RenderX (Fase 2)
Responsável por analisar imagens de marcenaria (prints 3D ou rascunhos em papel),
extrair anotações técnicas via OCR e gerar prompts fotorrealistas estruturados em JSON.
"""

import os
import json
from pathlib import Path
from typing import Union, Dict, Any
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

SYSTEM_INSTRUCTION = """
Você é um Arquiteto Sênior especialista em Marcenaria e Design de Interiores, atuando também como um Analista de OCR de alta precisão.
Sua missão é analisar imagens de projetos de marcenaria (que podem ser capturas de softwares 3D como SketchUp/Promob ou esboços manuais em papel com anotações) e extrair todas as informações técnicas para orientar um pipeline de renderização fotorrealista.

Suas responsabilidades:
1. Identificar com precisão se a imagem de entrada é:
   - "DIGITAL_3D": Modelo volumétrico digital, wireframe, print de software 3D CAD/BIM/SketchUp/Promob.
   - "PAPER_SKETCH": Rascunho, desenho à mão livre feito em papel, croqui manual com ou sem linhas de cota.
2. Realizar OCR minucioso de qualquer anotação textual ou numérica manuscrita ou impressa (dimensões, espessuras, materiais, acabamentos, tipos de madeira, cores, puxadores, ferragens).
3. Elaborar um prompt em inglês altamente descritivo e otimizado para motores de renderização fotorrealista (ex: Leonardo.ai / Stable Diffusion), preservando fielmente a geometria, volumes, divisões de nichos, gavetas e proporções da marcenaria apresentada.
   - O prompt deve incluir iluminação arquitetônica (studio lighting, soft ambient light), texturas realistas de madeira (wood grain, mdf, lacca), detalhes do ambiente interno e render ultra-realista 8k resolution, photorealistic, interior design photography.

FORMATO DE RESPOSTA OBRIGATÓRIO:
Você DEVE responder ESTRITAMENTE em formato JSON válido, contendo EXATAMENTE estas 3 chaves:
{
  "input_type": "DIGITAL_3D" ou "PAPER_SKETCH",
  "extracted_notes": "Resumo detalhado das medidas, tipos de acabamento, ferragens e anotações detectadas na imagem",
  "sd_prompt": "Prompt em inglês fotorrealista de alta qualidade detalhando a marcenaria, acabamento, textura e iluminação da cena"
}
"""


def analyze_carpentry_image(
    image_input: Union[str, Path, Image.Image],
    additional_instructions: str = "",
    api_key: str = None
) -> Dict[str, Any]:
    """
    Analisa uma imagem de projeto de marcenaria usando o Google Gemini 1.5 Pro.

    Args:
        image_input: Caminho da imagem (str/Path) ou objeto PIL.Image.Image.
        additional_instructions: Instruções ou preferências adicionais do usuário.
        api_key: Chave da API Gemini (opcional, busca em GEMINI_API_KEY se omitido).

    Returns:
        Dict com as chaves 'input_type', 'extracted_notes' e 'sd_prompt'.

    Raises:
        ValueError: Se a chave da API não estiver configurada ou a imagem for inválida.
        RuntimeError: Se ocorrer erro na chamada da API ou na decodificação do JSON.
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

    # 3. Configuração do Gemini
    genai.configure(api_key=gemini_key)

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.2
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=generation_config
    )

    # 4. Construção do prompt do usuário
    user_prompt = [
        "Analise esta imagem de projeto de marcenaria e extraia todas as informações no formato JSON especificado.",
        image
    ]

    if additional_instructions.strip():
        user_prompt.append(f"\nInstruções adicionais do marceneiro/designer: {additional_instructions.strip()}")

    # 5. Execução da inferência bimodal
    try:
        response = model.generate_content(user_prompt)
    except Exception as e:
        raise RuntimeError(f"Erro na comunicação com a API do Gemini: {e}") from e

    # 6. Parse e validação estrita do JSON retornado
    if not response.text:
        raise RuntimeError("A API do Gemini retornou uma resposta vazia.")

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"A resposta retornada não é um JSON válido: {response.text}") from e

    # Garante que as 3 chaves obrigatórias estejam presentes
    required_keys = ["input_type", "extracted_notes", "sd_prompt"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Chave obrigatória '{key}' ausente na resposta do Gemini: {data}")

    # Validação do valor de input_type
    if data["input_type"] not in ["DIGITAL_3D", "PAPER_SKETCH"]:
        # Fallback inteligente baseado em heurística caso a taxonomia divirja
        if "3D" in str(data["input_type"]).upper():
            data["input_type"] = "DIGITAL_3D"
        else:
            data["input_type"] = "PAPER_SKETCH"

    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso para teste manual:")
        print("  python vision_module.py <caminho_da_imagem> [instrucoes_adicionais]")
        sys.exit(1)

    test_img = sys.argv[1]
    extra_notes = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"[*] Analisando imagem: {test_img}")
    try:
        resultado = analyze_carpentry_image(test_img, additional_instructions=extra_notes)
        print("\n[+] Resultado estruturado recebido com sucesso:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    except Exception as err:
        print(f"\n[-] Erro durante análise: {err}")
        sys.exit(1)
