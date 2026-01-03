from lsp.lsp import *

class PythonLSP(LSP):
    """
    A simple Python Language Server Protocol implementation.
    Provides basic symbol indexing, go-to-definition, and completion.
    """

    def index_document(self, uri: str, text: str) -> None:
        # Index function and class definitions for go-to-definition and workspace symbols
        symbol_locations = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            match = re.match(r'^\s*(def|class)\s+(\w+)', line)
            if match:
                kind = 12 if match.group(1) == 'class' else 6  # SymbolKind.Class or Function
                name = match.group(2)
                pos = Position(line=i, character=line.find(name))
                loc = Location(uri=uri, range=(pos, pos))
                symbol_locations.append((name, kind, loc))
        self._symbol_index[uri] = [loc for _, _, loc in symbol_locations]
        self._documents[uri] = text
        self._symbols = symbol_locations

    def go_to_definition(self, uri: str, position: Position) -> Optional[Location]:
        # Find the word at the position
        text = self._documents.get(uri, "")
        lines = text.splitlines()
        if position.line >= len(lines):
            return None
        line = lines[position.line]
        words = re.findall(r'\w+', line)
        for word in words:
            idx = line.find(word)
            if idx <= position.character < idx + len(word):
                # Search for definition in index
                for name, _, loc in getattr(self, '_symbols', []):
                    if name == word:
                        return loc
        return None

    def find_references(self, uri: str, position: Position, include_declaration: bool = True) -> List[Location]:
        # Find all locations where the symbol at position is referenced
        text = self._documents.get(uri, "")
        lines = text.splitlines()
        if position.line >= len(lines):
            return []
        line = lines[position.line]
        words = re.findall(r'\w+', line)
        symbol = None
        for word in words:
            idx = line.find(word)
            if idx <= position.character < idx + len(word):
                symbol = word
                break
        if not symbol:
            return []
        locations = []
        for i, l in enumerate(lines):
            for match in re.finditer(r'\b{}\b'.format(re.escape(symbol)), l):
                pos = Position(line=i, character=match.start())
                locations.append(Location(uri=uri, range=(pos, pos)))
        return locations

    def autocomplete(self, uri: str, position: Position, prefix: str) -> List[CompletionItem]:
        # Suggest function and class names in the document that match the prefix
        items = []
        for name, kind, _ in getattr(self, '_symbols', []):
            if name.startswith(prefix):
                items.append(CompletionItem(label=name, kind=kind))