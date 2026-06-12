"""
AgentCode 的自绘聊天输入框。

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
            """保存提交时要交给 TUI 主应用的完整文本。"""

            self.text = text
            super().__init__()

    def __init__(self, text: str = "", *, placeholder: str = "", **kwargs: Any) -> None:
        """初始化输入缓冲、光标位置、历史导航和粘贴占位状态。"""

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
        """返回当前屏幕上显示的输入框文本。"""

        return self._text

    def submitted_text(self) -> str:
        """返回提交给模型的文本，会把大段粘贴占位符还原成原文。"""

        text = self._text
        for marker, original in self._paste_chunks.items():
            text = text.replace(marker, original)
        return text

    def add_history(self, text: str) -> None:
        """把成功提交的文本加入历史，并去掉连续重复项。"""

        if not text or (self._history and self._history[-1] == text):
            self._reset_history_navigation()
            return
        self._history.append(text)
        if len(self._history) > self.HISTORY_LIMIT:
            self._history = self._history[-self.HISTORY_LIMIT :]
        self._reset_history_navigation()

    def on_mount(self) -> None:
        """组件挂载后立即渲染初始占位符或文本。"""

        self._refresh_display()

    def on_focus(self) -> None:
        """获得焦点时显示硬件光标并同步 IME 候选框位置。"""

        self._set_terminal_cursor_visible(True)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def on_blur(self) -> None:
        """失去焦点时隐藏硬件光标并刷新输入框样式。"""

        self._set_terminal_cursor_visible(False)
        self._refresh_display()

    def focus(self, scroll_visible: bool = True) -> Self:
        """程序主动聚焦输入框时同步 Textual 状态和真实终端光标。"""

        # Textual 已经聚焦时不会再次触发 on_focus，因此重写 focus 保证程序聚焦也同步光标。
        focused = super().focus(scroll_visible)
        self._set_terminal_cursor_visible(not self.disabled)
        self._refresh_display()
        return focused

    def on_click(self) -> None:
        """鼠标点击输入框时获取焦点并同步光标位置。"""

        self.focus()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def focus_on_click(self) -> bool:
        """告诉 Textual 该组件点击后应获得焦点。"""

        return True

    def watch_disabled(self, disabled: bool) -> None:
        """输入框禁用状态变化时同步硬件光标和显示内容。"""

        self._set_terminal_cursor_visible(self.has_focus and not disabled)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def load_text(self, text: str) -> None:
        """用新文本替换当前输入，并清空粘贴占位和历史导航状态。"""

        self._set_text(text, clear_pastes=True)
        self._reset_history_navigation()
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def insert(self, text: str) -> None:
        """在当前光标处插入文本，并退出历史浏览状态。"""

        self._reset_history_navigation()
        self._insert_at_cursor(text)

    def on_paste(self, event: events.Paste) -> None:
        """接管 Textual 粘贴事件，以便大段内容折叠成占位符。"""

        event.prevent_default()
        event.stop()
        self._handle_paste(event.text)

    async def _on_key(self, event: events.Key) -> None:
        """处理键盘编辑、提交、历史导航和可打印字符输入。"""

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
        """设置内部文本缓冲和字符光标索引，可选择清空粘贴映射。"""

        self._text = text
        self._cursor_index = (
            len(text) if cursor_index is None else self._clamp_cursor(cursor_index)
        )
        self._keyboard_sequence = ""
        if clear_pastes:
            self._paste_chunks = {}

    def _insert_at_cursor(self, text: str) -> None:
        """按字符索引把文本插入缓冲区，并刷新显示和终端光标。"""

        if not text:
            return
        self._text = (
            self._text[: self._cursor_index] + text + self._text[self._cursor_index :]
        )
        self._cursor_index += len(text)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _delete_before_cursor(self) -> None:
        """删除光标左侧一个 Python 字符。"""

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
        """删除光标右侧一个 Python 字符。"""

        if self._cursor_index >= len(self._text):
            return
        self._reset_history_navigation()
        self._text = (
            self._text[: self._cursor_index] + self._text[self._cursor_index + 1 :]
        )
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _move_cursor(self, index: int) -> None:
        """移动字符光标到指定索引，并同步屏幕上的硬件光标。"""

        self._cursor_index = self._clamp_cursor(index)
        self._refresh_display()
        self._sync_terminal_cursor_position()

    def _clamp_cursor(self, index: int) -> int:
        """把目标字符索引限制在当前文本范围内。"""

        return min(max(index, 0), len(self._text))

    def _line_start_index(self, index: int) -> int:
        """返回指定索引所在行的起始字符索引。"""

        return self._text.rfind("\n", 0, index) + 1

    def _line_end_index(self, index: int) -> int:
        """返回指定索引所在行的结束字符索引。"""

        next_newline = self._text.find("\n", index)
        return len(self._text) if next_newline == -1 else next_newline

    def _cursor_is_on_first_line(self) -> bool:
        """判断当前光标前方是否没有换行。"""

        return "\n" not in self._text[: self._cursor_index]

    def _cursor_is_on_last_line(self) -> bool:
        """判断当前光标后方是否没有换行。"""

        return "\n" not in self._text[self._cursor_index :]

    def _history_previous(self) -> None:
        """向更早的提交历史移动，并保留进入历史前的草稿。"""

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
        """向更新的提交历史移动，到末尾时恢复原草稿。"""

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
        """退出历史浏览状态，丢弃临时草稿指针。"""

        self._history_index = None
        self._history_draft = ""

    def _handle_paste(self, text: str) -> None:
        """处理粘贴文本，大段粘贴折叠显示但提交时保留原文。"""

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
        """插入可打印字符，并兼容终端拆分发送的 CSI-u IME 序列。"""

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
        """提交前把尚未识别为 CSI-u 的缓存字符写回输入框。"""

        if self._keyboard_sequence:
            pending = self._keyboard_sequence
            self._keyboard_sequence = ""
            self.insert(pending)

    def _refresh_display(self) -> None:
        """重新渲染输入框文本并同步真实终端光标位置。"""

        self.update(self._render_input())
        self._sync_terminal_cursor_position()

    def _render_input(self) -> Text:
        """把内部文本缓冲渲染成带提示符的 Rich Text。"""

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
        """把 Textual 的硬件光标位置同步到当前字符光标所在终端 cell。"""

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
        """通过底层 driver 显示或隐藏真实终端光标。"""

        driver = getattr(self.app, "_driver", None)
        if driver is None or getattr(driver, "is_headless", False):
            return
        # Textual 进入 application mode 时会隐藏硬件光标；这里按输入焦点恢复可见性。
        driver.write(Control.show_cursor(visible).segment.text)
        driver.flush()

    def _cursor_line(self) -> tuple[int, str]:
        """返回光标所在行号和该行光标左侧文本。"""

        lines = self._text[: self._cursor_index].split("\n")
        return len(lines) - 1, lines[-1]


def _decode_csi_u_text(sequence: str) -> str | None:
    """把 Kitty/CSI-u 编码的 Unicode 码点序列还原为文本。"""

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
    """判断当前缓存是否仍可能是一个尚未完整到达的 CSI-u 序列。"""

    if len(sequence) > 128 or not sequence.startswith("["):
        return False
    return all(character.isdigit() or character in ";:" for character in sequence[1:])
