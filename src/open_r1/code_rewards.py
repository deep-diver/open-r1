import re
import json
import asyncio
import modal

my_image = modal.Image.from_registry(
    "ghcr.io/deep-diver/multipl-e:latest",
    add_python="3.11"
)

app = modal.App("rl-code-executor")

@app.function(image=my_image, timeout=480)
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
    test_cases = payload[2]

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
        rewards = 0

        for test_case in test_cases:
            stdin_data = test_case['input']
            output_data = test_case['output']

            result = subprocess.run(
                command,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "TMPDIR": "/tmp"},
            )

            if result.stdout.strip() == output_data.strip():
                rewards += 1

        return rewards / len(test_cases)

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        return 0.0


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer

def code_based_on_unittests_reward(completions, **kwargs) -> list[float]:
    language_info = kwargs["language"]
    print("language_info: ", language_info)

    code_snippets = [
        extract_code(completion[-1]["content"], language) for language, completion in zip(language_info, completions)
    ]
    print("code_snippets: \n", code_snippets)
    
    verification_info = kwargs["verification_info"]
        
    payloads = [
        (language, code, info["test_cases"]) 
        for code, info, language in zip(code_snippets, verification_info, language_info)
    ]

    async def run_modal_tasks(payloads):
        with modal.enable_output():
            # This is where your original async code goes
            async with app.run.aio():
                # Note the list comprehension uses 'async for'
                results = [r async for r in execute_code_for_rl.map.aio(payloads)]
            return results

    results = asyncio.run(run_modal_tasks(payloads))

    # 3. Your synchronous post-processing code continues here
    # results = [1 if result[4] else 0 for result in results]
    rewards = []
    for result in results:
        try:
            reward = float(result)
            rewards.append(reward)
        except:
            rewards.append(0.0)
    # return sum(rewards)
    print("rewards: ", rewards)
    return rewards
    
    # scripts = [
    #     evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
    #     for code, info in zip(code_snippets, verification_info)
    # ]    

    # rewards = run_async_from_sync(scripts, language_info)

    # except Exception as e:
    #     print(f"Error from Modal executor: {e}")
    #     rewards = [0.0] * len(completions)

    # return rewards

def unicoder_reward_fn(completions, **kwargs) -> list[float]:
    # current_epoch = kwargs["epoch"]
    # print(f"Current epoch: {current_epoch}")
    # allowed_labels = label_schedule(current_epoch)
    # print(f"Allowed labels: {allowed_labels}")
    # if allowed_labels is None:
    #     return [None] * len(completions)

    # for info in kwargs["verification_info"]:
    #     info["test_cases"] = [
    #         case for case in info["test_cases"]
    #         if (case.get("label") or "minimal") in allowed_labels
    #     ]

    rewards = code_based_on_unittests_reward(completions, **kwargs)
    return rewards
