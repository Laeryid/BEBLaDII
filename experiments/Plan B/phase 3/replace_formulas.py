import re

file_path = r"C:\Experiments\BEBLaDII\reports\plan_b_phase3_report.md"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    r"\$S\^\{d-1\}\$": "S^(d-1)",
    r"\$t \\in \[0,1\]\$": "t in [0, 1]",
    r"\$t=0\$": "t=0",
    r"\(\$x_0\$-prediction\)": "(x_0-prediction)",
    r"\$L = 1 - \\langle \\text\{DUS\}\(x_t, t\), x_0 \\rangle\$": "`L = 1 - <DUS(x_t, t), x_0>`",
    r"\(\$var\\_floor = 1.0 / \(2D\)\$\)": "`(var_floor = 1.0 / (2D))`",
    r"\$\\hat\{x\}_0\$": "`x_0_pred`",
    r"\$x_0\$": "`x_0`",
    r"\$t < 0.3\$": "t < 0.3",
    r"\$t > 0.7\$": "t > 0.7",
    r"\$t \\in \[0.1, 1.0\]\$": "t in [0.1, 1.0]",
    r"\$t=0.1\$": "t=0.1",
    r"\$t=0.3\$": "t=0.3",
    r"\$\\sim 0.98\$": "~0.98",
    r"\$t=0.4\$": "t=0.4",
    r"\$t=0.6\$": "t=0.6",
    r"\$t=0.5\$": "t=0.5",
    r"as \$t\$ decreases": "as `t` decreases",
    r"\$t \\ge 0.8\$": "t >= 0.8",
    r"\$x_t\$": "`x_t`",
    r"\$\\varepsilon\$": "pure noise (epsilon)"
}

for k, v in replacements.items():
    text = re.sub(k, v, text)

text = re.sub(r'\$([a-zA-Z0-9_=<\->., ]+)\$', r'`\1`', text)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Done")
