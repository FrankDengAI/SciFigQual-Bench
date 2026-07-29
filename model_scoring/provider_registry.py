"""Provider catalogue for SciFigQual scoring backends."""

from __future__ import annotations

from dataclasses import dataclass

import typer


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    api_key_env: str
    base_url_env: str | None
    default_base_url: str | None
    default_vision_model: str
    default_agent_llm_model: str
    paper_model: str
    category: str
    implemented: bool
    vision_capable: bool
    description_zh: str
    alt_api_key_envs: tuple[str, ...] = ()


# SIQA paper closed-source models: O3, GPT-5, GPT-4o, GPT-3.5, Gemini-2.5-Pro,
# Claude-sonnet-4.5, Doubao-Seed-2.0-pro
# Open-Source: GLM-4.6v, DeepSeek-VL2, LLaMA-3.2-90B-Vision, InternVL3.5, Qwen3-VL
PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI (GPT / o series)",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_API_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        default_vision_model="gpt-4o",
        default_agent_llm_model="gpt-4o",
        paper_model="GPT-5 / GPT-4o / o3 / GPT-3.5-turbo",
        category="closed",
        implemented=True,
        vision_capable=True,
        description_zh="Official OpenAI API — GPT-4o, o3, o4-mini and other multimodal models.",
    ),
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini",
        api_key_env="GEMINI_API_KEY",
        base_url_env=None,
        default_base_url=None,
        default_vision_model="gemini-2.5-pro",
        default_agent_llm_model="gemini-2.5-flash",
        paper_model="Gemini-2.5-Pro",
        category="closed",
        implemented=True,
        vision_capable=True,
        description_zh="Google AI Studio / Gemini API with native JSON schema output.",
    ),
    "claude": ProviderSpec(
        id="claude",
        label="Anthropic Claude",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env=None,
        default_base_url=None,
        default_vision_model="claude-sonnet-4-5",
        default_agent_llm_model="claude-sonnet-4-5",
        paper_model="Claude-sonnet-4.5",
        category="closed",
        implemented=True,
        vision_capable=True,
        description_zh="Anthropic Claude API with image input support.",
    ),
    "doubao": ProviderSpec(
        id="doubao",
        label="ByteDance Doubao",
        api_key_env="DOUBAO_API_KEY",
        base_url_env="DOUBAO_API_BASE_URL",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_vision_model="doubao-seed-2-0-pro-260215",
        default_agent_llm_model="doubao-seed-2-0-pro-260215",
        paper_model="Doubao-Seed-2.0-pro",
        category="closed",
        implemented=True,
        vision_capable=True,
        description_zh="Volcengine Ark OpenAI-compatible API; ARK_API_KEY also accepted.",
        alt_api_key_envs=("ARK_API_KEY",),
    ),
    "qwen": ProviderSpec(
        id="qwen",
        label="Alibaba Qwen",
        api_key_env="QWEN_API_KEY",
        base_url_env="QWEN_API_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_vision_model="qwen-vl-max",
        default_agent_llm_model="qwen-max",
        paper_model="Qwen3-VL-235B",
        category="open",
        implemented=True,
        vision_capable=True,
        description_zh="DashScope or MaaS OpenAI-compatible endpoint; Qwen-VL multimodal support.",
        alt_api_key_envs=("DASHSCOPE_API_KEY",),
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_API_BASE_URL",
        default_base_url="https://api.deepseek.com",
        default_vision_model="deepseek-chat",
        default_agent_llm_model="deepseek-chat",
        paper_model="DeepSeek-VL2",
        category="open",
        implemented=True,
        vision_capable=False,
        description_zh="Official DeepSeek API is text-only; use Qwen/Gemini/OpenAI or a vision gateway for figures.",
    ),
    "glm": ProviderSpec(
        id="glm",
        label="Zhipu GLM",
        api_key_env="GLM_API_KEY",
        base_url_env="GLM_API_BASE_URL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4/",
        default_vision_model="glm-4.6v",
        default_agent_llm_model="glm-4-plus",
        paper_model="GLM-4.6v",
        category="open",
        implemented=True,
        vision_capable=True,
        description_zh="Zhipu AI OpenAI-compatible API; ZHIPU_API_KEY also accepted.",
        alt_api_key_envs=("ZHIPU_API_KEY",),
    ),
    "internvl": ProviderSpec(
        id="internvl",
        label="InternVL (Shanghai AI Lab)",
        api_key_env="INTERNVL_API_KEY",
        base_url_env="INTERNVL_API_BASE_URL",
        default_base_url=None,
        default_vision_model="internvl3.5-241b",
        default_agent_llm_model="internvl3.5-241b",
        paper_model="InternVL3.5-241B",
        category="open",
        implemented=False,
        vision_capable=True,
        description_zh="SIQA paper baseline; self-host vLLM/OpenAI-compatible service and set Base URL.",
    ),
    "llama": ProviderSpec(
        id="llama",
        label="Meta LLaMA Vision",
        api_key_env="LLAMA_API_KEY",
        base_url_env="LLAMA_API_BASE_URL",
        default_base_url=None,
        default_vision_model="llama-3.2-90b-vision",
        default_agent_llm_model="llama-3.2-90b-vision",
        paper_model="LLaMA-3.2-90B-Vision",
        category="open",
        implemented=False,
        vision_capable=True,
        description_zh="SIQA paper baseline; use Together / Groq or other OpenAI-compatible endpoints.",
        alt_api_key_envs=("TOGETHER_API_KEY",),
    ),
}

UI_PROVIDER_ORDER: list[str] = [
    "openai",
    "gemini",
    "claude",
    "doubao",
    "qwen",
    "deepseek",
    "glm",
    "internvl",
    "llama",
]

SUPPORTED_PROVIDERS = frozenset(
    provider_id for provider_id, spec in PROVIDER_SPECS.items() if spec.implemented
)

DEFAULT_VISION_MODELS = {k: v.default_vision_model for k, v in PROVIDER_SPECS.items()}
DEFAULT_AGENT_LLM_MODELS = {k: v.default_agent_llm_model for k, v in PROVIDER_SPECS.items()}


def get_spec(provider: str) -> ProviderSpec:
    if provider not in PROVIDER_SPECS:
        raise typer.BadParameter(f"Unknown provider: {provider}")
    return PROVIDER_SPECS[provider]


def is_provider_implemented(provider: str) -> bool:
    return get_spec(provider).implemented


def is_provider_configured(provider: str, environ: dict[str, str] | None = None) -> bool:
    import os

    env = environ if environ is not None else os.environ
    spec = get_spec(provider)
    keys = (spec.api_key_env, *spec.alt_api_key_envs)
    return any(env.get(k, "").strip() for k in keys)


def validate_provider(provider: str, *, option_name: str = "provider") -> None:
    if provider not in SUPPORTED_PROVIDERS:
        names = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise typer.BadParameter(f"--{option_name} must be one of: {names}.")


def agent_llm_model(provider: str, vision_model: str) -> str:
    return DEFAULT_AGENT_LLM_MODELS.get(provider, vision_model)


def provider_deepseek_lacks_vision(provider: str, env: dict[str, str]) -> bool:
    if provider != "deepseek":
        return False
    import os

    base = (
        env.get("DEEPSEEK_API_BASE_URL", "")
        or os.environ.get("DEEPSEEK_API_BASE_URL", "")
        or "https://api.deepseek.com"
    ).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base in {"https://api.deepseek.com", ""}
