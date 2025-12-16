"""
IRH Desktop - Transparency Engine

The transparency engine is central to IRH's philosophy.
Every computation must explain itself with theoretical references.

Output Levels:
- INFO: High-level progress
- STEP: Individual operations
- DETAIL: Numerical specifics
- WHY: Plain-language explanations
- REF: Theoretical references
- WARN: Potential issues
- ERROR: Failures with context

Theoretical Foundation:
    IRH21.md - All computations reference specific equations
    
Author: Brandon D. McCrary
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from queue import Queue
from threading import Lock

logger = logging.getLogger(__name__)


class MessageLevel(Enum):
    """
    Transparency message levels.
    
    Each level serves a specific purpose in explaining computations.
    """
    INFO = auto()    # High-level progress
    STEP = auto()    # Individual operations
    DETAIL = auto()  # Numerical specifics
    WHY = auto()     # Plain-language explanation
    REF = auto()     # Theoretical reference
    WARN = auto()    # Potential issues
    ERROR = auto()   # Failures with context
    PASS = auto()    # Verification passed
    FAIL = auto()    # Verification failed


@dataclass
class TransparentMessage:
    """
    A message in the transparency system.
    
    Each message includes theoretical context and can be rendered
    in multiple formats (console, GUI, log file).
    
    Attributes
    ----------
    level : MessageLevel
        Message importance level
    message : str
        Main message text
    timestamp : datetime
        When the message was created
    equation : str, optional
        LaTeX equation if applicable
    reference : str, optional
        IRH21.md reference (e.g., "§1.2.3, Eq. 1.13")
    explanation : str, optional
        Plain-language explanation
    values : Dict[str, Any], optional
        Numerical values involved
    component : str, optional
        Which module/component generated this
    """
    level: MessageLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    equation: str = ""
    reference: str = ""
    explanation: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    component: str = ""
    
    def render_console(self, use_color: bool = True) -> str:
        """
        Render message for console output.
        
        Parameters
        ----------
        use_color : bool
            Whether to include ANSI color codes
            
        Returns
        -------
        str
            Formatted message string
        """
        # Color codes
        colors = {
            MessageLevel.INFO: "\033[36m",    # Cyan
            MessageLevel.STEP: "\033[33m",    # Yellow
            MessageLevel.DETAIL: "\033[37m",  # White
            MessageLevel.WHY: "\033[35m",     # Magenta
            MessageLevel.REF: "\033[34m",     # Blue
            MessageLevel.WARN: "\033[33;1m",  # Bold yellow
            MessageLevel.ERROR: "\033[31;1m", # Bold red
            MessageLevel.PASS: "\033[32m",    # Green
            MessageLevel.FAIL: "\033[31m",    # Red
        }
        reset = "\033[0m"
        
        # Build output
        time_str = self.timestamp.strftime("%H:%M:%S")
        level_str = f"[{self.level.name:6}]"
        
        if use_color:
            color = colors.get(self.level, "")
            output = f"{color}[{time_str}] {level_str} {self.message}{reset}"
        else:
            output = f"[{time_str}] {level_str} {self.message}"
        
        # Add reference
        if self.reference:
            ref_line = f"          REF   {self.reference}"
            if use_color:
                ref_line = f"\033[34m{ref_line}{reset}"
            output += f"\n{ref_line}"
        
        # Add equation
        if self.equation:
            eq_line = f"          DETAIL {self.equation}"
            output += f"\n{eq_line}"
        
        # Add values
        if self.values:
            for key, value in self.values.items():
                if isinstance(value, float):
                    val_str = f"{value:.12g}"
                else:
                    val_str = str(value)
                val_line = f"          DETAIL {key} = {val_str}"
                output += f"\n{val_line}"
        
        # Add explanation
        if self.explanation:
            exp_line = f"          WHY   {self.explanation}"
            if use_color:
                exp_line = f"\033[35m{exp_line}{reset}"
            output += f"\n{exp_line}"
        
        return output
    
    def render_html(self) -> str:
        """
        Render message for HTML/GUI display.
        
        Returns
        -------
        str
            HTML-formatted message
        """
        # Level-specific styling
        level_styles = {
            MessageLevel.INFO: "color: #00bcd4;",
            MessageLevel.STEP: "color: #ffc107;",
            MessageLevel.DETAIL: "color: #9e9e9e;",
            MessageLevel.WHY: "color: #e91e63;",
            MessageLevel.REF: "color: #2196f3;",
            MessageLevel.WARN: "color: #ff9800; font-weight: bold;",
            MessageLevel.ERROR: "color: #f44336; font-weight: bold;",
            MessageLevel.PASS: "color: #4caf50;",
            MessageLevel.FAIL: "color: #f44336;",
        }
        
        style = level_styles.get(self.level, "")
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        html = f'<div class="message" style="{style}">'
        html += f'<span class="time">[{time_str}]</span> '
        html += f'<span class="level">[{self.level.name}]</span> '
        html += f'<span class="text">{self.message}</span>'
        
        if self.reference:
            html += f'<div class="ref" style="margin-left: 2em; color: #2196f3;">'
            html += f'REF: {self.reference}</div>'
        
        if self.equation:
            html += f'<div class="eq" style="margin-left: 2em; font-family: monospace;">'
            html += f'{self.equation}</div>'
        
        if self.values:
            html += '<div class="values" style="margin-left: 2em;">'
            for key, value in self.values.items():
                if isinstance(value, float):
                    val_str = f"{value:.12g}"
                else:
                    val_str = str(value)
                html += f'<div>{key} = {val_str}</div>'
            html += '</div>'
        
        if self.explanation:
            html += f'<div class="why" style="margin-left: 2em; color: #e91e63;">'
            html += f'WHY: {self.explanation}</div>'
        
        html += '</div>'
        return html
    
    def render_log(self) -> str:
        """
        Render message for log file.
        
        Returns
        -------
        str
            Plain text log line
        """
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{time_str} | {self.level.name:6} | {self.message}"
        
        if self.reference:
            line += f" [REF: {self.reference}]"
        
        if self.values:
            vals = ", ".join(f"{k}={v}" for k, v in self.values.items())
            line += f" ({vals})"
        
        return line
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "equation": self.equation,
            "reference": self.reference,
            "explanation": self.explanation,
            "values": self.values,
            "component": self.component,
        }


class TransparencyEngine:
    """
    Engine for generating transparent, contextual computation output.
    
    The TransparencyEngine ensures all computations explain themselves
    with theoretical references to IRH21.md.
    
    Parameters
    ----------
    verbosity : int
        Verbosity level (1-5, default: 3)
    show_equations : bool
        Whether to show LaTeX equations
    show_explanations : bool
        Whether to show plain-language explanations
        
    Examples
    --------
    >>> engine = TransparencyEngine(verbosity=4)
    >>> engine.info("Starting RG flow computation", reference="§1.2")
    >>> engine.step("Computing β_λ at current point",
    ...             equation="β_λ = -2λ̃ + (9/8π²)λ̃²")
    >>> engine.detail("Result computed", values={"β_λ": 2.3e-14})
    >>> engine.why("The β-function measures how coupling runs with energy")
    >>> engine.passed("Fixed point condition satisfied: |β| < 10⁻¹⁰")
    
    Theoretical Foundation
    ----------------------
    This engine implements the transparency requirements from
    docs/DEB_PACKAGE_ROADMAP.md §4.2 "Transparent Output System"
    """
    
    def __init__(
        self,
        verbosity: int = 3,
        show_equations: bool = True,
        show_explanations: bool = True
    ):
        """
        Initialize the Transparency Engine.
        
        Parameters
        ----------
        verbosity : int
            Verbosity level (1=minimal, 5=maximum)
        show_equations : bool
            Show LaTeX equations
        show_explanations : bool
            Show explanatory text
        """
        self.verbosity = verbosity
        self.show_equations = show_equations
        self.show_explanations = show_explanations
        
        # Message history
        self.messages: List[TransparentMessage] = []
        self._lock = Lock()
        
        # Callbacks for real-time output
        self._callbacks: List[Callable[[TransparentMessage], None]] = []
        
        # Current computation context
        self._context_stack: List[str] = []
    
    def add_callback(self, callback: Callable[[TransparentMessage], None]) -> None:
        """
        Add a callback for real-time message handling.
        
        Parameters
        ----------
        callback : callable
            Function to call with each new message
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TransparentMessage], None]) -> None:
        """Remove a message callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def push_context(self, context: str) -> None:
        """
        Push a computation context onto the stack.
        
        Parameters
        ----------
        context : str
            Context description (e.g., "RG Flow Solver")
        """
        self._context_stack.append(context)
    
    def pop_context(self) -> Optional[str]:
        """Pop and return the current context."""
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def _emit(self, message: TransparentMessage) -> None:
        """
        Emit a message to all outputs.
        
        Parameters
        ----------
        message : TransparentMessage
            Message to emit
        """
        # Add context
        if self._context_stack:
            message.component = self._context_stack[-1]
        
        # Store in history
        with self._lock:
            self.messages.append(message)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Log to Python logger
        log_level = {
            MessageLevel.ERROR: logging.ERROR,
            MessageLevel.WARN: logging.WARNING,
            MessageLevel.INFO: logging.INFO,
        }.get(message.level, logging.DEBUG)
        
        logger.log(log_level, message.render_log())
    
    def _should_emit(self, level: MessageLevel) -> bool:
        """Check if a message level should be emitted based on verbosity."""
        # Verbosity mapping
        level_verbosity = {
            MessageLevel.ERROR: 1,
            MessageLevel.WARN: 1,
            MessageLevel.INFO: 2,
            MessageLevel.PASS: 2,
            MessageLevel.FAIL: 2,
            MessageLevel.STEP: 3,
            MessageLevel.REF: 3,
            MessageLevel.DETAIL: 4,
            MessageLevel.WHY: 5,
        }
        return level_verbosity.get(level, 3) <= self.verbosity
    
    def info(
        self,
        message: str,
        reference: str = "",
        **kwargs
    ) -> None:
        """
        Emit an INFO level message.
        
        Parameters
        ----------
        message : str
            Message text
        reference : str
            IRH21.md reference
        **kwargs
            Additional message attributes
        """
        if self._should_emit(MessageLevel.INFO):
            self._emit(TransparentMessage(
                level=MessageLevel.INFO,
                message=message,
                reference=reference,
                **kwargs
            ))
    
    def step(
        self,
        message: str,
        equation: str = "",
        reference: str = "",
        **kwargs
    ) -> None:
        """
        Emit a STEP level message for individual operations.
        
        Parameters
        ----------
        message : str
            Step description
        equation : str
            LaTeX equation being computed
        reference : str
            IRH21.md reference
        """
        if self._should_emit(MessageLevel.STEP):
            msg = TransparentMessage(
                level=MessageLevel.STEP,
                message=message,
                reference=reference,
                **kwargs
            )
            if self.show_equations and equation:
                msg.equation = equation
            self._emit(msg)
    
    def detail(
        self,
        message: str,
        values: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Emit a DETAIL level message with numerical values.
        
        Parameters
        ----------
        message : str
            Detail description
        values : dict
            Numerical values to display
        """
        if self._should_emit(MessageLevel.DETAIL):
            self._emit(TransparentMessage(
                level=MessageLevel.DETAIL,
                message=message,
                values=values or {},
                **kwargs
            ))
    
    def why(self, explanation: str, **kwargs) -> None:
        """
        Emit a WHY level message explaining the computation.
        
        Parameters
        ----------
        explanation : str
            Plain-language explanation
        """
        if self._should_emit(MessageLevel.WHY) and self.show_explanations:
            self._emit(TransparentMessage(
                level=MessageLevel.WHY,
                message="",
                explanation=explanation,
                **kwargs
            ))
    
    def ref(self, reference: str, description: str = "", **kwargs) -> None:
        """
        Emit a REF level message with theoretical reference.
        
        Parameters
        ----------
        reference : str
            IRH21.md reference (e.g., "§1.2.3, Eq. 1.13")
        description : str
            What the reference covers
        """
        if self._should_emit(MessageLevel.REF):
            self._emit(TransparentMessage(
                level=MessageLevel.REF,
                message=description,
                reference=reference,
                **kwargs
            ))
    
    def warn(self, message: str, **kwargs) -> None:
        """
        Emit a WARN level message.
        
        Parameters
        ----------
        message : str
            Warning message
        """
        if self._should_emit(MessageLevel.WARN):
            self._emit(TransparentMessage(
                level=MessageLevel.WARN,
                message=message,
                **kwargs
            ))
    
    def error(self, message: str, **kwargs) -> None:
        """
        Emit an ERROR level message.
        
        Parameters
        ----------
        message : str
            Error message
        """
        if self._should_emit(MessageLevel.ERROR):
            self._emit(TransparentMessage(
                level=MessageLevel.ERROR,
                message=message,
                **kwargs
            ))
    
    def passed(self, message: str, **kwargs) -> None:
        """
        Emit a PASS message for successful verification.
        
        Parameters
        ----------
        message : str
            What passed verification
        """
        if self._should_emit(MessageLevel.PASS):
            self._emit(TransparentMessage(
                level=MessageLevel.PASS,
                message=f"✓ {message}",
                **kwargs
            ))
    
    def failed(self, message: str, **kwargs) -> None:
        """
        Emit a FAIL message for failed verification.
        
        Parameters
        ----------
        message : str
            What failed verification
        """
        if self._should_emit(MessageLevel.FAIL):
            self._emit(TransparentMessage(
                level=MessageLevel.FAIL,
                message=f"✗ {message}",
                **kwargs
            ))
    
    def computation_start(
        self,
        name: str,
        reference: str = "",
        description: str = ""
    ) -> None:
        """
        Mark the start of a computation block.
        
        Parameters
        ----------
        name : str
            Computation name
        reference : str
            IRH21.md reference
        description : str
            What the computation does
        """
        self.push_context(name)
        self.info(f"Starting {name}", reference=reference)
        if description:
            self.why(description)
    
    def computation_end(
        self,
        success: bool = True,
        message: str = ""
    ) -> None:
        """
        Mark the end of a computation block.
        
        Parameters
        ----------
        success : bool
            Whether computation succeeded
        message : str
            Completion message
        """
        context = self.pop_context() or "computation"
        if success:
            self.passed(message or f"{context} completed successfully")
        else:
            self.failed(message or f"{context} failed")
    
    def get_history(self) -> List[TransparentMessage]:
        """Get all messages in history."""
        with self._lock:
            return list(self.messages)
    
    def clear_history(self) -> None:
        """Clear message history."""
        with self._lock:
            self.messages.clear()
    
    def export_log(self, path: str) -> bool:
        """
        Export message history to a log file.
        
        Parameters
        ----------
        path : str
            Output file path
            
        Returns
        -------
        bool
            True if export succeeded
        """
        try:
            with open(path, 'w') as f:
                f.write("IRH Computation Log\n")
                f.write("=" * 70 + "\n\n")
                
                for msg in self.get_history():
                    f.write(msg.render_log() + "\n")
            
            return True
        except Exception as e:
            logger.error(f"Failed to export log: {e}")
            return False
    
    def export_html(self, path: str) -> bool:
        """
        Export message history to HTML.
        
        Parameters
        ----------
        path : str
            Output file path
            
        Returns
        -------
        bool
            True if export succeeded
        """
        try:
            with open(path, 'w') as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <title>IRH Computation Log</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
        h1 { color: #569cd6; }
        .message { margin: 5px 0; padding: 5px; border-left: 3px solid #3a3a3a; }
        .time { color: #808080; }
        .level { font-weight: bold; }
    </style>
</head>
<body>
    <h1>IRH Computation Log</h1>
""")
                for msg in self.get_history():
                    f.write(msg.render_html() + "\n")
                
                f.write("</body></html>")
            
            return True
        except Exception as e:
            logger.error(f"Failed to export HTML: {e}")
            return False


# Global transparency engine instance
_global_engine: Optional[TransparencyEngine] = None


def get_transparency_engine() -> TransparencyEngine:
    """
    Get the global transparency engine instance.
    
    Returns
    -------
    TransparencyEngine
        Global engine instance
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = TransparencyEngine()
    return _global_engine


def set_transparency_engine(engine: TransparencyEngine) -> None:
    """
    Set the global transparency engine.
    
    Parameters
    ----------
    engine : TransparencyEngine
        Engine to use globally
    """
    global _global_engine
    _global_engine = engine
