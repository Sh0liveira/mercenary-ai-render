"""
Script de Validação e Autenticação de APIs - RenderX (Fase 1)
Testa a conectividade e autenticação com as APIs do Google Gemini e Leonardo.ai.
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()


def test_gemini_api(api_key: str) -> bool:
    """Valida a chave da API do Google Gemini listando modelos disponíveis."""
    print("\n[1/2] Testando conexão com Google Gemini API...")
    if not api_key or api_key == "sua_chave_gemini_aqui":
        print("  ❌ ERRO: GEMINI_API_KEY não configurada no arquivo .env.")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        # Tenta listar os modelos para confirmar autenticação
        models = genai.list_models()
        first_model = next(iter(models), None)

        if first_model:
            print("  ✅ Gemini API conectada com sucesso!")
            print(f"     Exemplo de modelo acessível: {first_model.name}")
            return True
        else:
            print("  ⚠️ Conexão efetuada, porém nenhum modelo foi retornado.")
            return True

    except Exception as e:
        print(f"  ❌ Falha na autenticação do Gemini API: {e}")
        return False


def test_leonardo_api(api_key: str) -> bool:
    """Valida a chave da API do Leonardo.ai consultando o endpoint de usuário (/me)."""
    print("\n[2/2] Testando conexão com Leonardo.ai API...")
    if not api_key or api_key == "sua_chave_leonardo_aqui":
        print("  ❌ ERRO: LEONARDO_API_KEY não configurada no arquivo .env.")
        return False

    url = "https://cloud.leonardo.ai/api/rest/v1/me"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            user_info = data.get("user_details", [{}])[0]
            username = user_info.get("user", {}).get("username", "Usuário")
            api_credit = user_info.get("apiCredit", "N/A")
            print("  ✅ Leonardo.ai API conectada com sucesso!")
            print(f"     Usuário: {username} | Créditos de API: {api_credit}")
            return True
        elif response.status_code == 401:
            print("  ❌ Falha de Autenticação (401 Unauthorized): Verifique sua LEONARDO_API_KEY.")
            return False
        else:
            print(f"  ❌ Erro na resposta da API Leonardo.ai (HTTP {response.status_code}): {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Erro de conexão com a API do Leonardo.ai: {e}")
        return False


def main():
    print("=" * 60)
    print("        RenderX - Teste de Autenticação de APIs (Fase 1)")
    print("=" * 60)

    gemini_key = os.getenv("GEMINI_API_KEY")
    leonardo_key = os.getenv("LEONARDO_API_KEY")

    gemini_ok = test_gemini_api(gemini_key)
    leonardo_ok = test_leonardo_api(leonardo_key)

    print("\n" + "=" * 60)
    print("                   RESUMO DOS TESTES")
    print("=" * 60)
    print(f" Google Gemini API : {'✅ SUCESSO' if gemini_ok else '❌ FALHOU'}")
    print(f" Leonardo.ai API   : {'✅ SUCESSO' if leonardo_ok else '❌ FALHOU'}")
    print("=" * 60)

    if gemini_ok and leonardo_ok:
        print("\n🎉 Todas as conexões foram bem-sucedidas! Ambiente pronto para a Fase 2.\n")
        sys.exit(0)
    else:
        print("\n⚠️ Uma ou mais conexões falharam. Atualize suas chaves no .env e tente novamente.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
