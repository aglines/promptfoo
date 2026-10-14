import re
 
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
 
 
def extract_last_number(text):
    matches = NUM.findall(text.replace("$", ""))
    return matches[-1].replace(",", "") if matches else None
 
 
def get_assert(output, context):
    expected = context["vars"]["expected"]
    got = extract_last_number(output)
 
    if got is None:
        return {"pass": False, "score": 0, "reason": "No number found in output"}
 
    passed = abs(float(got) - float(expected)) < 1e-6
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": f"Expected {expected}, got {got}",
    }
    
