"""TUI 自定义滚动条渲染器。

Textual 默认滚动条是硬块字符；这里用单列宽度和上下半块字符模拟 macOS
Chrome 那种无可见轨道、只有圆角 thumb 的悬浮滚动条。
"""

from __future__ import annotations

from math import ceil

from rich.color import Color
from rich.segment import Segment, Segments
from rich.style import Style
from textual.scrollbar import ScrollBar, ScrollBarRender


class PillScrollBarRender(ScrollBarRender):
    """把竖向滚动条渲染成无轨道的浅色胶囊形 thumb。"""

    @classmethod
    def render_bar(
        cls,
        size: int = 25,
        virtual_size: float = 50,
        window_size: float = 20,
        position: float = 0,
        thickness: int = 1,
        vertical: bool = True,
        back_color: Color = Color.parse("#444252"),
        bar_color: Color = Color.parse("#a7a7b3"),
    ) -> Segments:
        if not vertical:
            return super().render_bar(
                size,
                virtual_size,
                window_size,
                position,
                thickness,
                vertical,
                back_color,
                bar_color,
            )

        width = max(1, thickness)
        # Chrome/macOS 风格没有常驻可见轨道；空白段不设置背景色，让其融入内容区。
        track = Segment(" " * width)
        segments = [track] * max(size, 0)
        if not segments or not window_size or size == virtual_size:
            return Segments(segments, new_lines=True)

        thumb_size = _thumb_size(size, virtual_size, window_size)
        thumb_start = _thumb_start(
            size, virtual_size, window_size, position, thumb_size
        )
        thumb_end = min(size, thumb_start + thumb_size)

        for row in range(thumb_start, thumb_end):
            if thumb_size > 1 and row == thumb_start:
                text = _top_cap(width)
            elif thumb_size > 1 and row == thumb_end - 1:
                text = _bottom_cap(width)
            else:
                text = "█" * width
            segments[row] = Segment(
                text,
                Style(color=bar_color, meta={"@mouse.down": "grab"}),
            )

        return Segments(segments, new_lines=True)


def install_scrollbar_renderer() -> None:
    """安装全局滚动条渲染器，覆盖 Textual 默认硬块样式。"""

    ScrollBar.renderer = PillScrollBarRender


def _thumb_size(size: int, virtual_size: float, window_size: float) -> int:
    proportional = ceil(size * min(window_size / virtual_size, 1))
    minimum = 3 if size >= 3 else 1
    return max(minimum, min(size, proportional))


def _thumb_start(
    size: int,
    virtual_size: float,
    window_size: float,
    position: float,
    thumb_size: int,
) -> int:
    max_position = max(virtual_size - window_size, 1)
    max_start = max(size - thumb_size, 0)
    return max(0, min(max_start, round(max_start * position / max_position)))


def _top_cap(width: int) -> str:
    return "▄" * width


def _bottom_cap(width: int) -> str:
    return "▀" * width
