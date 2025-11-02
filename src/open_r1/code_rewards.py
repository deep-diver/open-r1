import re
import json
import asyncio
from .utils import is_e2b_available

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None
    
if is_modal_available():
    import modal
    
    my_image = modal.Image.from_registry(
        "ghcr.io/deep-diver/multipl-e:latest",
        add_python="3.11"
    )

    app = modal.App("rl-code-executor")

    @app.function(image=my_image, timeout=30)
    def execute_code_for_rl(payload: tuple) -> tuple:
        """
        payload:
        - (language, code)                          # stdin 없음 (과거 호환)
        - (language, code, stdin_string)            # stdin 포함

        반환: (stdout, stderr, returncode)
        """
        import os
        import subprocess

        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            return (None, "Invalid payload. Expected (language, code[, stdin]).", 1)

        # unpack with backward compatibility
        language = payload[0]
        code = payload[1]
        stdin_data = payload[2] if len(payload) >= 3 else ""
        output_data = payload[3] if len(payload) >= 4 else ""

        lang = str(language).lower().strip()
        if lang in ("javascript", "js"):
            lang = "javascript"
        elif lang in ("cpp", "c++"):
            lang = "cpp"

        try:
            if lang == "python":
                command = ["python3", "-c", code]

            elif lang == "go":
                src = "/tmp/run.go"
                with open(src, "w", encoding="utf-8") as f:
                    f.write(code)
                command = ["bash", "-lc", f"go run {src}"]

            elif lang == "rust":
                src = "/tmp/run.rs"
                bin_path = "/tmp/run_rust"
                with open(src, "w", encoding="utf-8") as f:
                    f.write(code)
                command = ["bash", "-lc", f"rustc {src} -O -o {bin_path} && {bin_path}"]

            elif lang == "javascript":
                src = "/tmp/run.js"
                with open(src, "w", encoding="utf-8") as f:
                    f.write(code)
                command = ["bash", "-lc", f"node {src}"]

            elif lang == "cpp":
                src = "/tmp/run.cpp"
                bin_path = "/tmp/run_cpp"
                with open(src, "w", encoding="utf-8") as f:
                    f.write(code)
                # compile; always show compiler output on stderr if any
                command = ["bash", "-lc", f"g++ -std=c++17 -O2 {src} -o {bin_path} 2>&1 | cat; test -x {bin_path} && {bin_path}"]

            else:
                return (None, f"Unsupported language: {language}", 1)

            # IMPORTANT: don't .strip() outputs; judges may require exact whitespace
            result = subprocess.run(
                command,
                input=stdin_data,       # pass stdin to the program
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "TMPDIR": "/tmp"},
            )

            return (
                command, 
                result.stdout, 
                result.stderr, 
                result.returncode, 
                result.stdout.strip() == output_data.strip()
            )

        except subprocess.TimeoutExpired:
            return (None, "Execution timed out", -1)
        except Exception as e:
            return (None, str(e), 1)


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer

def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards

async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script, language) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards

async def run_script(sbx: AsyncSandbox, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0
    except Exception as e:
        print(f"Error from E2B executor run_script: {e}")
        return 0.0

def code_based_on_unittests_reward(completions, **kwargs) -> list[float]:
    if not is_modal_available():
        raise ImportError(
            "Modal is not available and required for this reward function. Please install Modal with "
            "`pip install modal`."
        )

    # Returns a reward function that evaluates code snippets in a sandbox.
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        weights = {{
            "basic": 0.25,
            "medium": 0.25,
            "high": 0.25,
            "edge": 0.25
        }}

        passed_weight = 0.0
        total_weight = 0.0
        exec_timeout = 10

        for case in test_cases:
            label = case.get("label")
            if label is None: label = "minimal"
            weight = weights.get(label.strip(), 0.0)
            total_weight += weight

            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed_weight += weight

        # if total_weight == 0:
        #     return 0.0

        # weighted_success_rate = passed_weight / total_weight
        # return weighted_success_rate
        return passed_weight

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """        

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    
    language_info = kwargs["language"]
    verification_info = kwargs["verification_info"]
    
    payloads = [
        (code, info["text_cases"][0]['input'], language, info["test_cases"][0]['output'])
        for code, info, language in zip(code_snippets, verification_info, language_info)
    ]
    
    with modal.enable_output():
        # async context for Modal app
        async with app.run.aio():
            # gather results as they complete
            results = [r async for r in execute_code_for_rl.map.aio(payloads)]
            print(results)

    results = [1 if result[4] else 0 for result in results]
    return sum(results)
    
    # scripts = [
    #     evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
    #     for code, info in zip(code_snippets, verification_info)
    # ]    

    # rewards = run_async_from_sync(scripts, language_info)

    # except Exception as e:
    #     print(f"Error from Modal executor: {e}")
    #     rewards = [0.0] * len(completions)

    # return rewards

# Your curriculum logic
def label_schedule(epoch: int):
    # if step < 100:
    #     return None
    if epoch < 1:
        return ["basic"]
    elif epoch < 2:
        return ["basic", "medium"]
    elif epoch < 3:
        return ["basic", "medium", "high"]
    else:
        return ["basic", "medium", "high", "edge"]

def curriculum_aware_reward_fn(completions, **kwargs) -> list[float]:
    current_epoch = kwargs["epoch"]
    print(f"Current epoch: {current_epoch}")
    allowed_labels = label_schedule(current_epoch)
    print(f"Allowed labels: {allowed_labels}")
    # if allowed_labels is None:
    #     return [None] * len(completions)

    for info in kwargs["verification_info"]:
        info["test_cases"] = [
            case for case in info["test_cases"]
            if (case.get("label") or "minimal") in allowed_labels
        ]

    rewards = code_based_on_unittests_reward(completions, **kwargs)
    return rewards
