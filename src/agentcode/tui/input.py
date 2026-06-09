"""AgentCode 的自绘聊天输入框。

负责输入缓冲、闪烁光标、中文 IME/CSI-u 解码，以及同步真实终端光标给输入法。
"""

from __future__ import annotations

from typing import Any, Self

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.geometry import Offset
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static


class ChatInput(Static, can_focus=True):
    # TextArea 在部分终端/中文输入法下会出现宽字符渲染错位，因此聊天输入改为轻量自绘。
    FOCUS_ON_CLICK = True
    CURSOR_BLINK_SECONDS = 0.5
    CURSOR_CHARACTER = "│"

    class Submitted(Message):
        """用户按 Enter 提交输入框内容。"""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self, text: str = "", *, placeholder: str = "", **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._text = text
        self._keyboard_sequence = ""
        self._cursor_visible = False
        self._cursor_timer: Timer | None = None
        self.placeholder = placeholder

    @property
    def text(self) -> str:
        return self._text

    def on_mount(self) -> None:
        # 光标闪烁只在聚焦时运行，避免空闲时无意义刷新。
        self._cursor_timer = self.set_interval(
            self.CURSOR_BLINK_SECONDS,
            self._blink_cursor,
            pause=not self.has_focus,
        )
        self._refresh_display()

    def on_focus(self) -> None:
        self._cursor_visible = True
        if self._cursor_timer is not None:
            self._cursor_timer.resume()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def on_blur(self) -> None:
        self._cursor_visible = False
        if self._cursor_timer is not None:
            self._cursor_timer.pause()
        self._refresh_display()

    def focus(self, scroll_visible: bool = True) -> Self:
        # Textual 已经聚焦时不会再次触发 on_focus，因此重写 focus 保证程序聚焦也同步光标。
        focused = super().focus(scroll_visible)
        self._cursor_visible = True
        self._refresh_display()
        return focused

    def on_click(self) -> None:
        self.focus()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def focus_on_click(self) -> bool:
        return True

    def watch_disabled(self, disabled: bool) -> None:
        if disabled:
            self._cursor_visible = False
            if self._cursor_timer is not None:
                self._cursor_timer.pause()
        elif self.has_focus:
            self._cursor_visible = True
            if self._cursor_timer is not None:
                self._cursor_timer.resume()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def load_text(self, text: str) -> None:
        self._text = text
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def insert(self, text: str) -> None:
        self._text += text
        self._refresh_display()
        self._sync_terminal_cursor_position()

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
            self.post_message(self.Submitted(self.text))
            return

        if event.key == "backspace":
            event.prevent_default()
            event.stop()
            if self._keyboard_sequence:
                self._keyboard_sequence = ""
                return
            self._text = self._text[:-1]
            self._refresh_display()
            self._sync_terminal_cursor_position()
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

    def _blink_cursor(self) -> None:
        if not self.has_focus or self.disabled:
            self._cursor_visible = False
            return
        self._cursor_visible = not self._cursor_visible
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _refresh_display(self) -> None:
        self.update(self._render_input())
        self._sync_terminal_cursor_position()

    def _render_input(self) -> Text:
        rendered = Text()
        cursor = (
            self.CURSOR_CHARACTER
            if self._cursor_visible and self.has_focus and not self.disabled
            else ""
        )
        if not self._text:
            # 空态永远预留一格光标槽，避免光标闪烁或首次聚焦时 placeholder 横向抖动。
            rendered.append("❯ ", style="bold cyan")
            rendered.append(cursor or " ")
            rendered.append(self.placeholder, style="dim")
            return rendered

        lines = f"{self._text}{cursor}".split("\n")
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

    def _cursor_line(self) -> tuple[int, str]:
        lines = self._text.split("\n")
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
