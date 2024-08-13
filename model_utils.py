import ollama
from tqdm import tqdm

def pull_model(name: str) -> None:
    current_digest, bars = "", {}
    for progress in ollama.pull(name, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()
        if not digest:
            print(progress.get("status"))
            continue
        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True)
        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)
        current_digest = digest

def is_model_available_locally(model_name: str) -> bool:
    if model_name == "OpenAI":
        # Skip checking for OpenAI model in Ollama
        return True
    try:
        ollama.show(model_name)
        return True
    except ollama.ResponseError:
        return False

def get_list_of_models() -> list[str]:
    return [model["name"] for model in ollama.list()["models"]]

def check_model_availability(model_name: str) -> None:
    if model_name == "OpenAI":
        return  # Skip availability check for OpenAI model
    try:
        available = is_model_available_locally(model_name)
    except Exception:
        raise Exception("Unable to communicate with the Ollama service")
    if not available:
        try:
            pull_model(model_name)
        except Exception:
            raise Exception(f"Unable to find model '{model_name}', please check the name and try again.")
