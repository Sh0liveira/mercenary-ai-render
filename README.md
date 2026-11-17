<div align="center">
  <h1>🪵 WoodAI Studio <em>(V2)</em></h1>
  <p><strong>Transformando plantas técnicas de marcenaria em renderizações fotorealistas com IA Autônoma.</strong></p>

  <!-- Badges -->
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Gemini_AI-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/Leonardo.ai-SDXL-orange?style=for-the-badge" alt="Leonardo.ai" />
</div>

<br>

## 📖 Sobre o Projeto

O **WoodAI Studio** é um sistema automatizado para o setor de marcenaria e design de interiores. O desafio principal deste projeto foi resolver o **"Gargalo do ControlNet"**: motores de difusão de imagem falham ao interpretar plantas técnicas 2D porque leem linhas de cota (ex: "156cm") e textos como se fossem geometria 3D, gerando alucinações (como gavetas flutuantes ou textos esculpidos na madeira).

Para resolver isso, desenvolvemos um **Agente Autônomo** que orquestra um pipeline híbrido de **Visão Computacional clássica (OpenCV)** para limpeza de imagem, **LLMs Multimodais (Google Gemini)** para extração semântica, e **Modelos de Difusão (Leonardo Kino XL + ControlNet)** para a renderização fotorrealista.

---

## ✨ Principais Funcionalidades

- 🤖 **Pipeline de IA Autônomo:** Um loop inteligente que avalia a própria imagem gerada.
- 👁️ **Limpeza de Imagem Determinística:** Uso de OpenCV para apagar cotas e textos antes da geração, evitando "alucinações" do ControlNet.
- 📐 **Tradução Semântico-Geométrica:** LLMs convertem medidas exatas (ex: `156x80cm`) em descrições de proporção em inglês (ex: `wide horizontal proportion`), blindando o prompt do motor 3D.
- ⚖️ **QA Judge (Juiz de Qualidade):** Um modelo Vision rigoroso que reprova a imagem caso não atinja um *score* de fotorrealismo ou se detectar resíduos de textos na madeira.
- 📉 **Decaimento Dinâmico de Peso (Weight Decay):** Se o Juiz reprovar o fotorrealismo, o sistema automaticamente reduz a força estrutural do ControlNet (ex: de `0.50` para `0.35`) e tenta novamente sozinho.

---

## 🏗️ Arquitetura do Sistema

O projeto adota uma arquitetura modularizada, baseada em responsabilidades únicas:

1. `preprocess_module.py`: Responsável por aplicar filtros morfológicos e algoritmos de *inpainting* (OpenCV) para limpar rabiscos e textos da planta original.
2. `vision_module.py`: Utiliza Gemini OCR para extrair os dados estruturais e montar prompts otimizados para o motor SDXL.
3. `render_module.py`: Comunica-se com a API do Leonardo.ai, configurando o ControlNet (Edge) e aplicando o modelo fotográfico Kino XL.
4. `judge_module.py`: Atua como o Diretor de Arte, avaliando a fidelidade da estrutura gerada e o fotorrealismo.
5. `app.py`: A interface Streamlit e o coração do Agente Autônomo, orquestrando o loop de tentativas.

---

## 🚀 Como Rodar Localmente

### Pré-requisitos
- Python 3.10 ou superior
- Chaves de API do [Google Gemini](https://aistudio.google.com/) e [Leonardo.ai](https://leonardo.ai/)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/woodai-studio.git
cd woodai-studio
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as Variáveis de Ambiente:
Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
```env
GEMINI_API_KEY=sua_chave_gemini_aqui
LEONARDO_API_KEY=sua_chave_leonardo_aqui
```

5. Inicie a aplicação:
```bash
streamlit run app.py
```

---

## 💻 Exemplo de Código: O Loop Autônomo

Um dos diferenciais deste projeto é o loop de tentativa e decaimento de peso (implementado no `app.py`). Caso a IA gere uma imagem ruim, o sistema se autocorrige:

```python
# Trecho simplificado do fluxo de Agente Autônomo
controlnet_weight = 0.50
max_attempts = 3

for attempt in range(max_attempts):
    # 1. Gera a imagem
    render_path = render_carpentry_design(
        clean_img_path, prompt, controlnet_weight
    )
    
    # 2. O Juiz avalia a qualidade
    evaluation = judge_render(render_path, structural_data)
    
    if evaluation.is_approved:
        display_success(render_path)
        break
    else:
        # 3. Decaimento de peso e autocorreção do prompt
        controlnet_weight -= 0.15
        prompt = update_prompt_with_judge_feedback(
            prompt, evaluation.prompt_corrections
        )
        print(f"Tentativa falhou. Reduzindo peso para {controlnet_weight}...")
```

---

## 🔮 Próximos Passos (Roadmap)
- [x] Desacoplamento da geometria de textos (OpenCV).
- [x] Implementação de Juiz de Qualidade (Vision QA).
- [ ] Suporte a múltiplas câmeras e ângulos na mesma requisição.
- [ ] Deploy na nuvem utilizando Docker e Google Cloud Run.

---
<div align="center">
  <i>Desenvolvido por <strong>Sergio</strong> — Engenharia de Software & IA Aplicada</i>
</div>
