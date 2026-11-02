# Records docs/demo/evaluate.gif from a real run: `vhs docs/demo/evaluate.tape` from the repo root.
Output docs/demo/evaluate.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 360
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `python -m neural_word_embeddings evaluate` Enter
Sleep 5s
