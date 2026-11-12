"""
Script de Teste Integrado E2E - RenderX (Pipeline Fase 2 + Fase 3)
Conecta os dois cérebros:
1. vision_module (Google Gemini OCR & Análise Estruturada)
2. render_module (Leonardo.ai SDXL Image-to-Image & Renderização Fotorrealista)
"""

import sys
import json
from pathlib import Path

# Importação dos módulos das Fases 2 e 3
from vision_module import analyze_carpentry_image
from render_module import render_carpentry_design


def run_integrated_pipeline(image_path: str = "imgs/images.jpeg"):
    print("=" * 70)
    print("       RENDERX - TESTE INTEGRADO DO PIPELINE (FASE 2 + FASE 3)")
    print("=" * 70)

    path = Path(image_path)
    if not path.is_file():
        print(f"❌ Imagem de teste não encontrada em: {path}")
        print("Certifique-se de que o caminho está correto ou passe como argumento.")
        sys.exit(1)

    print(f"\n[ETAPA 1/2] Executando Módulo de Visão & OCR (Gemini)...")
    print(f"  -> Imagem de entrada: {path}")

    try:
        vision_result = analyze_carpentry_image(
            image_input=path,
            additional_instructions="Acabamento premium em carvalho claro, portas com fechamento suave, iluminação quente e fotorrealismo 8k."
        )
        print("\n✅ [Cérebro 1 - Gemini] Análise concluída com sucesso!")
        print(f"   Tipo detectado : {vision_result['input_type']}")
        print(f"   Notas extraídas: {vision_result['extracted_notes'][:120]}...")
        print(f"   Prompt SD      : {vision_result['sd_prompt'][:120]}...")
    except Exception as e:
        print(f"\n❌ [Cérebro 1 - Gemini] Falha na análise de visão: {e}")
        sys.exit(1)

    print(f"\n[ETAPA 2/2] Executando Motor de Renderização (Leonardo.ai SDXL Image-to-Image)...")
    try:
        rendered_file = render_carpentry_design(
            image_path=path,
            vision_data=vision_result,
            output_dir="output",
            strength_type="High"
        )
        print("\n🎉 [Pipeline Completo] SUCESSO ABSOLUTO!")
        print(f"   Imagem fotorrealista gerada e salva em: {rendered_file}\n")
    except Exception as e:
        print(f"\n⚠️ [Cérebro 2 - Leonardo.ai] Interrupção no motor de renderização:")
        print(f"   {e}")
        print("\n[!] Dica para resolução caso o erro seja de autenticação (401/402):")
        print("    Acesse https://cloud.leonardo.ai, verifique se sua chave possui créditos")
        print("    ativos no plano da API de Desenvolvedor (Developer API Platform).")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "imgs/images.jpeg"
    run_integrated_pipeline(target)
