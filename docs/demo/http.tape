# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m neural_word_embeddings serve --port 18318 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18318/similar -d '{"word": "king", "k": 3}' | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18318/analogy -d '{"a": "king", "b": "queen", "c": "prince", "k": 2}' | jq -c` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
