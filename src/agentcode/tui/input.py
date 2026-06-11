"""AgentCode 的自绘聊天输入框。

负责输入缓冲、光标编辑、历史输入、粘贴折叠、中文 IME/CSI-u 解码，
以及同步真实终端光标给输入法。
"""

from __future__ import annotations

from typing import Any, Self

from rich.cells import cell_len
from rich.control import Control
from rich.text import Text
from textual import events
from textual.geometry import Offset
from textual.message import Message
from textual.widgets import Static


class ChatInput(Static, can_focus=True):
    # TextArea 在部分终端/中文输入法下会出现宽字符渲染错位，因此聊天输入改为轻量自绘。
    FOCUS_ON_CLICK = True
    HISTORY_LIMIT = 100
    LARGE_PASTE_LINES = 10
    LARGE_PASTE_CHARS = 1000

    class Submitted(Message):
        """用户按 Enter 提交输入框内容。"""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self, text: str = "", *, placeholder: str = "", **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._text = text
        self._cursor_index = len(text)
        self._keyboard_sequence = ""
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._paste_chunks: dict[str, str] = {}
        self._paste_counter = 0
        self.placeholder = placeholder

    @property
    def text(self) -> str:
        return self._text

    def submitted_text(self) -> str:
        text = self._text
        for marker, original in self._paste_chunks.items():
            text = text.replace(marker, original)
        return text

    def add_history(self, text: str) -> None:
        if not text or (self._history and self._history[-1] == text):
            self._reset_history_navigation()
            return
        self._history.append(text)
        if len(self._history) > self.HISTORY_LIMIT:
            self._history = self._history[-self.HISTORY_LIMIT :]
        self._reset_history_navigation()

    def on_mount(self) -> None:
        self._refresh_display()

    def on_focus(self) -> None:
        self._set_terminal_cursor_visible(True)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def on_blur(self) -> None:
        self._set_terminal_cursor_visible(False)
        self._refresh_display()

    def focus(self, scroll_visible: bool = True) -> Self:
        # Textual 已经聚焦时不会再次触发 on_focus，因此重写 focus 保证程序聚焦也同步光标。
        focused = super().focus(scroll_visible)
        self._set_terminal_cursor_visible(not self.disabled)
        self._refresh_display()
        return focused

    def on_click(self) -> None:
        self.focus()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def focus_on_click(self) -> bool:
        return True

    def watch_disabled(self, disabled: bool) -> None:
        self._set_terminal_cursor_visible(self.has_focus and not disabled)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def load_text(self, text: str) -> None:
        self._set_text(text, clear_pastes=True)
        self._reset_history_navigation()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def insert(self, text: str) -> None:
        self._reset_history_navigation()
        self._insert_at_cursor(text)

    def on_paste(self, event: events.Paste) -> None:
        event.prevent_default()
        event.stop()
        self._handle_paste(event.text)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"alt+enter", "meta+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self._flush_keyboard_sequence()
            self.post_message(self.Submitted(self.submitted_text()))
            return

        if event.key == "backspace":
            event.prevent_default()
            event.stop()
            if self._keyboard_sequence:
                self._keyboard_sequence = ""
                return
            self._delete_before_cursor()
            return

        if event.key == "delete":
            event.prevent_default()
            event.stop()
            self._delete_after_cursor()
            return

        if event.key in {"left", "ctrl+b"}:
            event.prevent_default()
            event.stop()
            self._move_cursor(self._cursor_index - 1)
            return

        if event.key in {"right", "ctrl+f"}:
            event.prevent_default()
            event.stop()
            self._move_cursor(self._cursor_index + 1)
            return

        if event.key in {"home", "ctrl+a"}:
            event.prevent_default()
            event.stop()
            self._move_cursor(self._line_start_index(self._cursor_index))
            return

        if event.key in {"end", "ctrl+e"}:
            event.prevent_default()
            event.stop()
            self._move_cursor(self._line_end_index(self._cursor_index))
            return

        if event.key == "up" and self._cursor_is_on_first_line():
            event.prevent_default()
            event.stop()
            self._history_previous()
            return

        if event.key == "down" and self._cursor_is_on_last_line():
            event.prevent_default()
            event.stop()
            self._history_next()
            return

        if event.key == "ctrl+u":
            event.prevent_default()
            event.stop()
            self._keyboard_sequence = ""
            self.load_text("")
            return

        if event.is_printable and event.character:
            event.prevent_default()
            event.stop()
            self._insert_printable(event.character)

    def _set_text(
        self,
        text: str,
        cursor_index: int | None = None,
        *,
        clear_pastes: bool = False,
    ) -> None:
        self._text = text
        self._cursor_index = (
            len(text) if cursor_index is None else self._clamp_cursor(cursor_index)
        )
        self._keyboard_sequence = ""
        if clear_pastes:
            self._paste_chunks = {}

    def _insert_at_cursor(self, text: str) -> None:
        if not text:
            return
        self._text = (
            self._text[: self._cursor_index] + text + self._text[self._cursor_index :]
        )
        self._cursor_index += len(text)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _delete_before_cursor(self) -> None:
        if self._cursor_index <= 0:
            return
        self._reset_history_navigation()
        self._text = (
            self._text[: self._cursor_index - 1] + self._text[self._cursor_index :]
        )
        self._cursor_index -= 1
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _delete_after_cursor(self) -> None:
        if self._cursor_index >= len(self._text):
            return
        self._reset_history_navigation()
        self._text = (
            self._text[: self._cursor_index] + self._text[self._cursor_index + 1 :]
        )
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _move_cursor(self, index: int) -> None:
        self._cursor_index = self._clamp_cursor(index)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _clamp_cursor(self, index: int) -> int:
        return min(max(index, 0), len(self._text))

    def _line_start_index(self, index: int) -> int:
        return self._text.rfind("\n", 0, index) + 1

    def _line_end_index(self, index: int) -> int:
        next_newline = self._text.find("\n", index)
        return len(self._text) if next_newline == -1 else next_newline

    def _cursor_is_on_first_line(self) -> bool:
        return "\n" not in self._text[: self._cursor_index]

    def _cursor_is_on_last_line(self) -> bool:
        return "\n" not in self._text[self._cursor_index :]

    def _history_previous(self) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._history_draft = self._text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        self._set_text(self._history[self._history_index], clear_pastes=True)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            text = self._history[self._history_index]
        else:
            text = self._history_draft
            self._reset_history_navigation()
        self._set_text(text, clear_pastes=True)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _reset_history_navigation(self) -> None:
        self._history_index = None
        self._history_draft = ""

    def _handle_paste(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized:
            return

        line_count = normalized.count("\n") + 1
        if (
            line_count > self.LARGE_PASTE_LINES
            or len(normalized) > self.LARGE_PASTE_CHARS
        ):
            self._paste_counter += 1
            if line_count > 1:
                marker = f"[paste #{self._paste_counter} +{line_count} lines]"
            else:
                marker = f"[paste #{self._paste_counter} +{len(normalized)} chars]"
            self._paste_chunks[marker] = normalized
            self.insert(marker)
            return

        self.insert(normalized)

    def _insert_printable(self, text: str) -> None:
        decoded = _decode_csi_u_text(text)
        if decoded is not None:
            self.insert(decoded)
            return

        for character in text:
            if self._keyboard_sequence:
                # 一些终端会把中文 IME 结果拆成 CSI-u 序列逐字符发来，需要临时拼起来。
                self._keyboard_sequence += character
                decoded = _decode_csi_u_text(self._keyboard_sequence)
                if decoded is not None:
                    self._keyboard_sequence = ""
                    self.insert(decoded)
                elif not _is_csi_u_prefix(self._keyboard_sequence):
                    pending = self._keyboard_sequence
                    self._keyboard_sequence = ""
                    self.insert(pending)
                continue

            if character == "[":
                # CSI-u 去掉 ESC 后通常以 "[" 开头；先缓存，等后续字符判断是否完整序列。
                self._keyboard_sequence = character
                continue

            self.insert(character)

    def _flush_keyboard_sequence(self) -> None:
        if self._keyboard_sequence:
            pending = self._keyboard_sequence
            self._keyboard_sequence = ""
            self.insert(pending)

    def _refresh_display(self) -> None:
        self.update(self._render_input())
        self._sync_terminal_cursor_position()

    def _render_input(self) -> Text:
        rendered = Text()
        if not self._text:
            rendered.append("❯ ", style="bold cyan")
            rendered.append(self.placeholder, style="dim")
            return rendered

        lines = self._text.split("\n")
        for index, line in enumerate(lines):
            if index:
                rendered.append("\n")
            rendered.append("❯ " if index == 0 else "  ", style="bold cyan")
            rendered.append(line)
        return rendered

    def _sync_terminal_cursor_position(self) -> None:
        if not self.has_focus or self.disabled:
            return

        line_index, line_text = self._cursor_line()
        prompt_width = 2
        # 真实终端光标用于 IME 候选框定位；中文要按 cell 宽度计算，不能用 len()。
        self.app.cursor_position = Offset(
            self.content_region.x + prompt_width + cell_len(line_text),
            self.content_region.y + line_index,
        )

    def _set_terminal_cursor_visible(self, visible: bool) -> None:
        driver = getattr(self.app, "_driver", None)
        if driver is None or getattr(driver, "is_headless", False):
            return
        # Textual 进入 application mode 时会隐藏硬件光标；这里按输入焦点恢复可见性。
        driver.write(Control.show_cursor(visible).segment.text)
        driver.flush()

    def _cursor_line(self) -> tuple[int, str]:
        lines = self._text[: self._cursor_index].split("\n")
        return len(lines) - 1, lines[-1]


def _decode_csi_u_text(sequence: str) -> str | None:
    # Kitty/CSI-u 键盘协议可能把 IME 文本编码为 "[32;;20320:21834u"。
    if sequence.startswith("\x1b"):
        sequence = sequence[1:]
    if not sequence.startswith("[") or not sequence.endswith("u"):
        return None

    fields = sequence[1:-1].split(";")
    if len(fields) < 3:
        return None

    text_points = fields[-1]
    if not text_points:
        return None

    try:
        return "".join(chr(int(point)) for point in text_points.split(":"))
    except (ValueError, OverflowError):
        return None


def _is_csi_u_prefix(sequence: str) -> bool:
    if len(sequence) > 128 or not sequence.startswith("["):
        return False
    return all(character.isdigit() or character in ";:" for character in sequence[1:])
