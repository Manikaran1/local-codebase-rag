"""Tree-sitter adapter for extracting Python symbols into domain objects."""

from __future__ import annotations

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from codebase_rag.domain.identity import symbol_id_for
from codebase_rag.domain.models import SourceFile, SourceSpan, Symbol
from codebase_rag.parsing.base import CodeParser

_PYTHON_LANGUAGE = Language(tree_sitter_python.language())
_DEFINITION_NODES = {"class_definition", "function_definition"}


class PythonParser(CodeParser):
    """Extract classes, functions, and methods without exposing syntax nodes."""

    def supports(self, source: SourceFile) -> bool:
        return source.relative_path.suffix.lower() == ".py" or (
            source.language is not None and source.language.casefold() == "python"
        )

    def parse(self, source: SourceFile, text: str) -> list[Symbol]:
        symbols, _ = self.parse_with_diagnostics(source, text)
        return symbols

    def parse_with_diagnostics(
        self,
        source: SourceFile,
        text: str,
    ) -> tuple[list[Symbol], tuple[str, ...]]:
        """Return recoverable symbols and a safe diagnostic for malformed source."""
        source_bytes = text.encode("utf-8")
        tree = Parser(_PYTHON_LANGUAGE).parse(source_bytes)
        symbols: list[Symbol] = []
        self._visit(tree.root_node, source, source_bytes, None, symbols)
        diagnostics = (
            ("Python syntax errors were detected; malformed definition subtrees were omitted.",)
            if tree.root_node.has_error
            else ()
        )
        return symbols, diagnostics

    def _visit(
        self,
        node: Node,
        source: SourceFile,
        source_bytes: bytes,
        parent: Symbol | None,
        symbols: list[Symbol],
    ) -> None:
        if node.type == "ERROR":
            return

        if node.type == "decorated_definition":
            definition = next(
                (child for child in node.named_children if child.type in _DEFINITION_NODES),
                None,
            )
            if definition is not None:
                if node.has_error or definition.has_error:
                    return
                symbol = self._make_symbol(
                    definition,
                    span_node=node,
                    source=source,
                    source_bytes=source_bytes,
                    parent=parent,
                )
                symbols.append(symbol)
                for child in definition.named_children:
                    self._visit(child, source, source_bytes, symbol, symbols)
                return

        if node.type in _DEFINITION_NODES:
            if node.has_error:
                return
            symbol = self._make_symbol(
                node,
                span_node=node,
                source=source,
                source_bytes=source_bytes,
                parent=parent,
            )
            symbols.append(symbol)
            for child in node.named_children:
                self._visit(child, source, source_bytes, symbol, symbols)
            return

        for child in node.named_children:
            self._visit(child, source, source_bytes, parent, symbols)

    def _make_symbol(
        self,
        definition: Node,
        *,
        span_node: Node,
        source: SourceFile,
        source_bytes: bytes,
        parent: Symbol | None,
    ) -> Symbol:
        name_node = definition.child_by_field_name("name")
        name = (
            source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node is not None
            else "<anonymous>"
        )
        kind = (
            "class"
            if definition.type == "class_definition"
            else "method"
            if parent is not None and parent.kind == "class"
            else "function"
        )
        qualified_name = f"{parent.qualified_name}.{name}" if parent is not None else name
        start_line = span_node.start_point.row + 1
        end_row, end_column = definition.end_point
        end_line = end_row if end_column == 0 else end_row + 1
        span = SourceSpan(start_line=start_line, end_line=max(start_line, end_line))
        body = definition.child_by_field_name("body")
        signature_end = body.start_byte if body is not None else definition.end_byte
        signature = source_bytes[definition.start_byte:signature_end].decode("utf-8").strip()
        return Symbol(
            id=symbol_id_for(source.id, qualified_name, kind, span),
            file_id=source.id,
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            language="Python",
            span=span,
            parent_id=parent.id if parent is not None else None,
            signature=signature or None,
        )
