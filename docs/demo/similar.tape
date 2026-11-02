# Records docs/demo/similar.gif from a real run: `vhs docs/demo/similar.tape` from the repo root.
Output docs/demo/similar.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 460
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m neural_word_embeddings similar queen` Enter
Sleep 5s
Type `python -m neural_word_embeddings analogy king queen prince` Enter
Sleep 6s
