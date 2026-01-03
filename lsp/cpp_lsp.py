from typing import Optional
from lsp.lsp import LSP, Position, Location, CompletionItem, HoverInfo, Diagnostic, SymbolInformation, SignatureHelp

class CppLSP(LSP):
    """
    A simple C++ Language Server Protocol implementation.
    This is a minimal, illustrative implementation and does not use a real C++ parser.
    """
    def index_document(self, uri: str, text: str) -> None:
        # Naive symbol index: collect function and class names
        symbols = []
        for match in re.finditer(r'\b(class|struct|void|int|float|double|char|bool)\s+(\w+)\s*\(', text):
            name = match.group(2)
            line = text[:match.start()].count('\n')
            start = Position(line=line, character=match.start() - text.rfind('\n', 0, match.start()) - 1)
            end = Position(line=line, character=start.character + len(name))
            loc = Location(uri=uri, range=(start, end))
            symbols.append(loc)
            self._symbol_index.setdefault(name, []).append(loc)

    def go_to_definition(self, uri: str, position: Position) -> Optional[Location]:
        # Naive: find the word at position and look up in symbol index
        word = self._get_word_at(uri, position)
        locs = self._symbol_index.get(word)
        return locs[0] if locs else None

    def find_references(self, uri: str, position: Position, include_declaration: bool = True) -> List[Location]:
        word = self._get_word_at(uri, position)
        return self._symbol_index.get(word, [])

    def autocomplete(self, uri: str, position: Position, prefix: str) -> List[CompletionItem]:
        # Suggest all known symbols starting with prefix
        items = []
        for name in self._symbol_index:
            if name.startswith(prefix):
                items.append(CompletionItem(label=name))
        return items

    def hover(self, uri: str, position: Position) -> Optional[HoverInfo]:
        word = self._get_word_at(uri, position)
        if word in self._symbol_index:
            return HoverInfo(contents=f"C++ symbol: {word}")
        return None

    def diagnostics(self, uri: str) -> List[Diagnostic]:
        # Dummy: no diagnostics
        return []

    def workspace_symbols(self, query: str) -> List[SymbolInformation]:
        results = []
        for name, locs in self._symbol_index.items():
            if query in name:
                for loc in locs:
                    results.append(SymbolInformation(name=name, kind=12, location=loc))  # 12: SymbolKind.Function
        return results

    def signature_help(self, uri: str, position: Position) -> Optional[SignatureHelp]:
        # Dummy: no signature help
        return None

    def _get_word_at(self, uri: str, position: Position) -> str:
        # Naive word extraction at position
        text = self._documents.get(uri, "")
        lines = text.splitlines()
        if position.line >= len(lines):
            return ""
        line = lines[position.line]
        if position.character > len(line):
            return ""
        # Find word boundaries around the character
        left = right = position.character
        while left > 0 and (line[left-1].isalnum() or line[left-1] == '_'):
            left -= 1
        while right < len(line) and (line[right].isalnum() or line[right] == '_'):
            right += 1
        return line[left:right]