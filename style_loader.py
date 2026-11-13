"""
Carregador da Biblioteca de Estilos em YAML - RenderX (Fase 2.5)
Responsável por carregar regras visuais (camera, environment, lighting,
quality_tags e negative) separadas da estrutura física do móvel.
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml

STYLES_DIR = Path(__file__).resolve().parent / "styles"

# Mapeamento opcional de aliases comuns (PT-BR / variações) para os nomes de arquivos YAML
FURNITURE_TYPE_ALIASES = {
    "guarda-roupa": "wardrobe",
    "guarda_roupa": "wardrobe",
    "roupeiro": "wardrobe",
    "closet": "wardrobe",
    "armario de cozinha": "kitchen_cabinet",
    "armário de cozinha": "kitchen_cabinet",
    "cozinha": "kitchen_cabinet",
    "gabinete de cozinha": "kitchen_cabinet",
}


def _normalize_furniture_type(furniture_type: str) -> str:
    """Normaliza o tipo de móvel para localizar o arquivo YAML correspondente."""
    clean = (furniture_type or "").strip().lower()
    if clean in FURNITURE_TYPE_ALIASES:
        return FURNITURE_TYPE_ALIASES[clean]
    return clean.replace(" ", "_").replace("-", "_")


def load_style(furniture_type: str, variant: str = "modern_minimal") -> Dict[str, Any]:
    """
    Carrega as configurações de estilo a partir do YAML correspondente ao tipo de móvel.
    Caso o arquivo específico do móvel não exista, realiza fallback seguro para `default.yaml`.

    Args:
        furniture_type: Categoria do móvel (ex: "wardrobe", "kitchen_cabinet", "aparador").
        variant: Nome da variante de estilo (padrão: "modern_minimal").

    Returns:
        Dicionário contendo as chaves: camera, environment, lighting, quality_tags e negative.

    Raises:
        FileNotFoundError: Se nem o arquivo específico nem o `default.yaml` forem encontrados.
        ValueError: Se o YAML estiver malformado ou não possuir variantes válidas.
    """
    normalized_type = _normalize_furniture_type(furniture_type)
    target_file = STYLES_DIR / f"{normalized_type}.yaml"
    default_file = STYLES_DIR / "default.yaml"

    yaml_path = target_file if target_file.exists() else default_file
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Arquivo de estilo não encontrado: '{target_file}' e fallback '{default_file}' ausente."
        )

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    variants = data.get("variants", {})
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"Nenhuma variante válida encontrada em '{yaml_path}'.")

    # Busca a variante solicitada; se não existir no arquivo específico, tenta fallback seguro
    if variant in variants:
        return dict(variants[variant])

    if "modern_minimal" in variants:
        return dict(variants["modern_minimal"])

    # Fallback para a primeira variante disponível
    first_variant = next(iter(variants.values()))
    return dict(first_variant)


def list_available_styles() -> Dict[str, List[str]]:
    """Lista os tipos de móveis e suas respectivas variantes disponíveis em `styles/`."""
    catalog: Dict[str, List[str]] = {}
    if not STYLES_DIR.exists():
        return catalog

    for yaml_file in sorted(STYLES_DIR.glob("*.yaml")):
        with yaml_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        ftype = data.get("furniture_type", yaml_file.stem)
        variants = list((data.get("variants") or {}).keys())
        catalog[ftype] = variants

    return catalog
