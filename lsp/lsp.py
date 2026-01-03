from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass
import re
import re


# Data classes for LSP types
dataclass
class Position:
    line: int
    character: int

@dataclass
class Location:
    uri: str
    range: tuple[Position, Position]

@dataclass
class CompletionItem:
    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None
    documentation: Optional[str] = None

@dataclass
class HoverInfo:
    contents: Any  # Can be string or markdown
    range: Optional[tuple[Position, Position]] = None

@dataclass
class Diagnostic:
    range: tuple[Position, Position]
    severity: int
    code: Optional[str]
    message: str

@dataclass
class SymbolInformation:
    name: str
    kind: int
    location: Location

@dataclass
class SignatureHelp:
    signatures: List[Any]
    active_signature: int
    active_parameter: int


class LSP(ABC):
    """
    Abstract base class defining Language Server Protocol (LSP) interface.
    Implementations must provide protocol-specific logic for code intelligence features.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self._initialized = False
        self._documents: dict[str, str] = {}
        self._symbol_index: dict[str, List[Location]] = {}

    def load_workspace(self) -> None:
        """
        Load and index workspace files for fast symbol lookup and navigation.
        """
        # default: optionally override to parse files
        pass

    def add_document(self, uri: str, text: str) -> None:
        """Add or update a document in the LSP server."""
        self._documents[uri] = text
        self.index_document(uri, text)

    def index_document(self, uri: str, text: str) -> None:
        """
        Parse and index the document for symbols and definitions.
        """
        # default: override with parser logic
        pass

    def initialize(self) -> None:
        """
        Initialize the LSP server (handshake, capabilities, etc.).
        """
        if not self._initialized:
            self._initialized = True
            self.load_workspace()

    def shutdown(self) -> None:
        """
        Clean up any resources before shutdown.
        """
        self._initialized = False

    @abstractmethod
    def go_to_definition(self, uri: str, position: Position) -> Optional[Location]:
        """
        Return the Location of the definition for the symbol at the given position.
        """
        raise NotImplementedError

    @abstractmethod
    def find_references(self, uri: str, position: Position, include_declaration: bool = True) -> List[Location]:
        """
        Return a list of Locations where the symbol is referenced in the workspace.
        """
        raise NotImplementedError

    @abstractmethod
    def autocomplete(self, uri: str, position: Position, prefix: str) -> List[CompletionItem]:
        """
        Provide completion items for the given prefix at position.
        """
        raise NotImplementedError

    @abstractmethod
    def hover(self, uri: str, position: Position) -> Optional[HoverInfo]:
        """
        Return hover information (e.g., documentation) for the symbol at the position.
        """
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, uri: str) -> List[Diagnostic]:
        """
        Analyze the document and return any diagnostics (errors, warnings).
        """
        raise NotImplementedError

    @abstractmethod
    def workspace_symbols(self, query: str) -> List[SymbolInformation]:
        """
        Search for symbols matching the query across the workspace.
        """
        raise NotImplementedError

    @abstractmethod
    def signature_help(self, uri: str, position: Position) -> Optional[SignatureHelp]:
        """
        Provide signature help (parameter info) at the given position.
        """
        raise NotImplementedError

    def send_notification(self, method: str, params: Any) -> None:
        """Send an LSP notification message to the client."""
        # override to integrate with JSONRPC or transport layer
        pass

    def send_request(self, method: str, params: Any) -> Any:
        """Send an LSP request and return the response."""
        # override to integrate with JSONRPC or transport layer
        pass