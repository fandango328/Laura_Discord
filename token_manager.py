#!/usr/bin/env python3

"""
TokenManager for SmartAss Voice Assistant
Last Updated: 2025-03-16 20:08:25

Purpose:
    Manages token usage, costs, and tool state for the SmartAss voice assistant,
    with specific handling of tool-related overheads and model-specific pricing.

Key Features:
    - Token counting and cost calculation for different Claude models
    - Tool usage and definition overhead tracking
    - Simplified binary tool state with auto-disable after 3 non-tool queries
    - Voice-optimized command phrases for reliable tool state management
    - Session-based cost accumulation
"""
import os
import json
import glob
import traceback
from config_og import CHAT_LOG_MAX_TOKENS, CHAT_LOG_RECOVERY_TOKENS, CHAT_LOG_DIR, SYSTEM_PROMPT, ANTHROPIC_MODEL 
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple, Optional, Any, List, Union
from laura_tools import AVAILABLE_TOOLS, get_tool_by_name, get_tools_by_category

class TokenManager:
    """
    Manages token counting, cost calculation, and tool usage authorization for SmartAss.
    
    Features:
    - Binary tool state (enabled/disabled)
    - Auto-disable tools after 3 consecutive non-tool queries
    - Phonetically optimized voice commands for tool state changes
    - Mixed command resolution (prioritizing last mentioned command)
    - Token and cost tracking for API interactions
    """

    # Model-specific costs per million tokens (MTok)
    MODEL_COSTS = {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00,     # $3.00 per 1M input tokens = $0.000003 per token
            "output": 15.00    # $15.00 per 1M output tokens = $0.000015 per token
        },
        "claude-3-7-sonnet-20240229": {
            "input": 3.00,     # $3.00 per 1M input tokens = $0.000003 per token
            "output": 15.00    # $15.00 per 1M output tokens = $0.000015 per token
        },
        "claude-3-opus-20240229": {
            "input": 15.00,    # $15.00 per 1M input tokens = $0.000015 per token
            "output": 75.00    # $75.00 per 1M output tokens = $0.000075 per token
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.80,     # $0.80 per 1M input tokens = $0.0000008 per token
            "output": 4.00     # $4.00 per 1M output tokens = $0.000004 per token
        }
    }

    # Token overheads for different operations
    TOOL_COSTS = {
        "definition_overhead": 2600,  # Base tokens for full tool definitions JSON
        "usage_overhead": {
            "auto": 346,     # When tool_choice="auto"
            "any": 313,      # When tool_choice allows any tool
            "tool": 313,     # Specific tool usage
            "none": 0        # No tools used
        }
    }

    # Voice-optimized tool enabling phrases
    # These phrases were specifically selected to have distinctive phonetic patterns
    # that are less likely to be confused by voice transcription systems like VOSK
    TOOL_ENABLE_PHRASES = {
        # Primary commands (most distinctive)
        "tools activate", "launch toolkit", "begin assistance", "enable tool use",
        "start tools", "enable assistant", "tools online", "enable tools",
        
        # Additional distinctive commands
        "assistant power up", "toolkit online", "helper mode active",
        "utilities on", "activate functions", "tools ready",
        
        # Short commands with distinctive sounds
        "tools on", "toolkit on", "functions on",
        
        # Commands with unique phonetic patterns
        "wake up tools", "prepare toolkit", "bring tools online"
    }

    # Voice-optimized tool disabling phrases
    # Selected for clear phonetic distinction from enabling phrases and from
    # common conversation patterns to minimize false positives/negatives
    TOOL_DISABLE_PHRASES = {
        # Primary commands (most distinctive)
        "tools offline", "end toolkit", "close assistant",
        "stop tools", "disable assistant", "conversation only",
        
        # Additional distinctive commands
        "assistant power down", "toolkit offline", "helper mode inactive",
        "utilities off", "deactivate functions", "tools away",
        
        # Short commands with distinctive sounds
        "tools off", "toolkit off", "functions off",
        
        # Commands with unique phonetic patterns
        "sleep tools", "dismiss toolkit", "take tools offline"
    }

    # Tool category keywords for contextual tool detection
    TOOL_CATEGORY_KEYWORDS = {
        'EMAIL': ['email', 'mail', 'send', 'write', 'compose', 'draft'],
        'CALENDAR': ['calendar', 'schedule', 'event', 'meeting', 'appointment'],
        'TASK': ['task', 'todo', 'reminder', 'checklist', 'to-do'],
        'UTILITY': ['time', 'location', 'calibrate', 'voice', 'settings'],
        'CONTACT': ['contact', 'person', 'people', 'address', 'phone']
    }

    def __init__(self, anthropic_client):
        """
        Initialize TokenManager with streamlined session tracking.
        
        Args:
            anthropic_client: Anthropic API client instance
            
        Raises:
            TypeError: If anthropic_client is None
        """
        # CHANGE: Simplified client validation - removed MODEL_COSTS check since we're using native token counting
        if anthropic_client is None:
            raise TypeError("anthropic_client cannot be None")
            
        # KEEP: Core client and model settings
        self.anthropic_client = anthropic_client
        self.query_model = ANTHROPIC_MODEL
        self.tool_model = "claude-3-5-sonnet-20241022"

        # KEEP: Session state tracking
        self.session_active = False
        self.session_start_time = None

        
        self.haiku_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': Decimal('0.00')
        }
        
        self.sonnet_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_definition_tokens': 0,
            'cost': Decimal('0.00'),
            'tools_initialized': False
        }

        # Update session tracker to include model info
        self.session = {
            'current_model': 'claude-3-5-haiku-20241022',
            'history_tokens': 0,
            'total_cost': Decimal('0.00'),
            'tools_enabled': False,
        }
              
        # KEEP: Tool state management
        self.tools_enabled = False
        self.tools_used_in_session = False
        self.consecutive_non_tool_queries = 0
        self.tool_disable_threshold = 3

    def start_interaction(self):
        """
        Start tracking a new interaction.
        Should be called at the beginning of each user interaction.
        """
        if not self.session_active:
            self.start_session()
            
        print("\n=== Token Log ===")
        return {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "tools_enabled": self.tools_enabled,
            "session_active": self.session_active
        }

    def log_error(self, error_message: str):
        """
        Log an error that occurred during interaction.
        
        Args:
            error_message: The error message to log
            
        Returns:
            Dict with error logging status
        """
        try:
            timestamp = datetime.now().isoformat()
            error_log_dir = os.path.join(CHAT_LOG_DIR, "errors")
            os.makedirs(error_log_dir, exist_ok=True)
            
            log_file = os.path.join(error_log_dir, f"error_log_{datetime.now().strftime('%Y-%m-%d')}.json")
            
            error_entry = {
                "timestamp": timestamp,
                "error": error_message,
                "session_state": {
                    "tools_enabled": self.tools_enabled,
                    "tools_used": self.tools_used_in_session,
                    "consecutive_non_tool_queries": self.consecutive_non_tool_queries
                }
            }
            
            # Load existing log if it exists
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Error log file {log_file} was corrupted, starting fresh")
                    
            # Append new error and write back
            if not isinstance(existing_logs, list):
                existing_logs = []
            existing_logs.append(error_entry)
            
            # Write atomically using a temporary file
            temp_file = f"{log_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
            os.replace(temp_file, log_file)
            
            print(f"\nError logged: {error_message}")
            return {
                "status": "logged",
                "timestamp": timestamp,
                "location": log_file
            }
            
        except Exception as e:
            print(f"Failed to log error: {str(e)}")
            return {
                "status": "failed",
                "reason": str(e)
            }


    def prepare_messages_for_token_count(self, current_query: str, chat_log: list, system_prompt: str = None) -> tuple:
        formatted_messages = []

        # Handle chat log messages
        if chat_log and isinstance(chat_log, list):
            for msg in chat_log:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg['role'],
                        "content": [{
                            "type": "text",
                            "text": str(msg['content'])
                        }]
                    })

        # Add current query
        formatted_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": str(current_query)}]
        })

        system_content = (system_prompt or SYSTEM_PROMPT).strip()
        return formatted_messages, system_content

    def count_message_tokens(self, current_query: str, chat_log: list = None, system_prompt: str = None) -> int:
        """Count input tokens using Anthropic's official endpoint."""
        try:
            messages, system_content = self.prepare_messages_for_token_count(current_query, chat_log, system_prompt)

            # Log the prepared messages and system content
            #print(f"DEBUG: Prepared messages for token count: {messages}")
            #print(f"DEBUG: Prepared system content: {system_content}")

            count_result = self.anthropic_client.messages.count_tokens(
                model=self.query_model,
                messages=messages,
                system=system_content
            )

            # Log the API response
            #print(f"DEBUG: Anthropic API response for input tokens: {count_result}")

            input_tokens = count_result.input_tokens if hasattr(count_result, 'input_tokens') else 0

            if self.tools_enabled:
                self.sonnet_tracker['input_tokens'] += input_tokens
            else:
                self.haiku_tracker['input_tokens'] += input_tokens

            return input_tokens

        except Exception as e:
            print(f"ERROR in token counting: {str(e)}")
            traceback.print_exc()
            return 0
        
    def count_output_tokens(self, response_text: str) -> int:
        """
        Estimates output tokens for cost tracking.
        
        Args:
            response_text: The text response from Claude
            
        Returns:
            int: Estimated token count for cost calculation
        """
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            print(f"ERROR: Response text is not a string: {response_text}")
            return 0

        # Log the response text
        #print(f"DEBUG: Response text for output token estimation: {response_text}")

        char_count = len(response_text)
        estimated_tokens = char_count // 4

        print(f"\nOutput Token Estimation:")
        print(f"Characters: {char_count}")
        print(f"Estimated Tokens: {estimated_tokens}")

        return estimated_tokens

    def start_session(self):
        self.session_active = True
        self.session_start_time = datetime.now()
        
        self.haiku_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': Decimal('0.00')
        }
        
        self.sonnet_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'tool_definition_tokens': 0,
            'cost': Decimal('0.00'),
            'tools_initialized': False
        }
        
        self.session = {
            'current_model': 'claude-3-5-haiku-20241022',
            'history_tokens': 0,
            'total_cost': Decimal('0.00'),
            'tools_enabled': False
        }
        
        self.tools_enabled = False
        self.tools_used_in_session = False
        self.consecutive_non_tool_queries = 0
        
        print(f"Token tracking session started at {self.session_start_time}")
        return {
            "status": "started",
            "timestamp": self.session_start_time.isoformat()
        }

    def display_token_usage(self, query_tokens: int = None):
        print("\n┌─ Token Usage Report " + "─" * 40)
        
        if query_tokens:
            print(f"│ Current Query Tokens: {query_tokens:,}")
            print("│" + "─" * 50)
            
        print("│ Haiku Usage:")
        print(f"│   Input Tokens:  {self.haiku_tracker['input_tokens']:,}")
        print(f"│   Output Tokens: {self.haiku_tracker['output_tokens']:,}")
        print(f"│   Cost:         ${self.haiku_tracker['cost']:.4f}")
        
        print("│\n│ Sonnet Usage:")
        print(f"│   Input Tokens:  {self.sonnet_tracker['input_tokens']:,}")
        print(f"│   Output Tokens: {self.sonnet_tracker['output_tokens']:,}")
        print(f"│   Tool Def Tokens: {self.sonnet_tracker['tool_definition_tokens']:,}")
        print(f"│   Cost:         ${self.sonnet_tracker['cost']:.4f}")
        
        print("│\n│ Session Totals:")
        total_cost = self.haiku_tracker['cost'] + self.sonnet_tracker['cost']
        print(f"│   Total Cost:    ${total_cost:.4f}")
        
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            print(f"│   Duration:      {duration.total_seconds():.1f}s")
        
        print("└" + "─" * 50)

    def update_session_costs(self, input_tokens: int, response_text: str, is_tool_use: bool = False):
        try:
            tracker = self.sonnet_tracker if is_tool_use else self.haiku_tracker
            model = self.tool_model if is_tool_use else self.query_model

            # Log the input tokens and response text
            print(f" Input tokens: {input_tokens}")
            print(f" Response text: {response_text}")

            output_tokens = self.count_output_tokens(response_text)  # Ensure correct response text

            print(f" Updating session costs - input_tokens: {input_tokens}, output_tokens: {output_tokens}")

            current_input = max(0, input_tokens)
            current_output = max(0, output_tokens)

            if is_tool_use and not self.sonnet_tracker['tools_initialized']:
                current_input += self.TOOL_COSTS['definition_overhead']
                self.sonnet_tracker['tool_definition_tokens'] = self.TOOL_COSTS['definition_overhead']
                self.sonnet_tracker['tools_initialized'] = True

            input_cost = self.calculate_token_cost(model, "input", current_input)
            output_cost = self.calculate_token_cost(model, "output", current_output)

            tracker['input_tokens'] += current_input
            tracker['output_tokens'] += current_output
            tracker['cost'] += input_cost + output_cost

            self.session['total_cost'] += input_cost + output_cost
            self.session['history_tokens'] += current_input + current_output

            print(f" Session costs updated - input_cost: {input_cost}, output_cost: {output_cost}")

            self.display_token_usage(current_input + current_output)

        except Exception as e:
            print(f"Error updating session costs: {e}")

    def calculate_token_cost(self, model: str, token_type: str, token_count: int) -> Decimal:
        try:
            if token_count < 0:
                print(f"Warning: Negative token count ({token_count}) adjusted to 0")
                token_count = 0
            
            if model not in self.MODEL_COSTS:
                print(f"Warning: Unknown model '{model}', using claude-3-5-sonnet-20241022 pricing")
                model = "claude-3-5-sonnet-20241022"
                
            if token_type not in self.MODEL_COSTS[model]:
                print(f"Warning: Unknown token type '{token_type}' for model '{model}', using 'input' pricing")
                token_type = "input"
                
            per_million_rate = self.MODEL_COSTS[model][token_type]
            if not isinstance(per_million_rate, Decimal):
                per_million_rate = Decimal(str(per_million_rate))
                
            return (per_million_rate / Decimal('1000000')) * Decimal(str(token_count))
            
        except Exception as e:
            print(f"Error calculating token cost: {e}")
            return Decimal('0')

    def get_session_summary(self) -> Dict[str, Any]:
        session_duration = datetime.now() - self.session_start_time
        minutes = session_duration.total_seconds() / 60
        
        haiku = self.haiku_tracker
        sonnet = self.sonnet_tracker
        
        return {
            "session_duration_minutes": round(minutes, 2),
            "haiku_stats": {
                "input_tokens": haiku['input_tokens'],
                "output_tokens": haiku['output_tokens'],
                "total_cost": float(haiku['cost'])
            },
            "sonnet_stats": {
                "input_tokens": sonnet['input_tokens'],
                "output_tokens": sonnet['output_tokens'],
                "tool_definition_tokens": sonnet['tool_definition_tokens'],
                "total_cost": float(sonnet['cost'])
            },
            "session_totals": {
                "total_tokens": (haiku['input_tokens'] + haiku['output_tokens'] + 
                               sonnet['input_tokens'] + sonnet['output_tokens']),
                "total_cost_usd": float(haiku['cost'] + sonnet['cost'])
            },
            "tools_currently_enabled": self.tools_enabled,
            "tools_used_in_session": self.tools_used_in_session
        }

    def enable_tools(self, query: str = None) -> Dict[str, Any]:
        self.tools_enabled = True
        self.consecutive_non_tool_queries = 0
        
        return {
            "state": "enabled",
            "message": "Tools are now enabled. What would you like help with?",
            "tools_active": True
        }

    def disable_tools(self, reason: str = "manual") -> Dict[str, Any]:
        was_enabled = self.tools_enabled
        self.tools_enabled = False
        self.consecutive_non_tool_queries = 0
        
        message = "Tools are now disabled." if was_enabled else "Tools are already disabled."
        if reason == "auto":
            message = "Tools have been automatically disabled after 3 queries without tool usage."
            
        return {
            "state": "disabled",
            "message": message,
            "tools_active": False
        }

    def record_tool_usage(self, tool_name: str) -> Dict[str, Any]:
        self.tools_used_in_session = True
        self.consecutive_non_tool_queries = 0
        
        print(f"Tool usage recorded: {tool_name}")
        #print(f"Non-tool query counter reset to 0")
        
        return {
            "status": "recorded",
            "tool": tool_name,
            "reset_counter": True
        }

    def track_query_completion(self, used_tool: bool = False) -> Dict[str, Any]:
        if not self.tools_enabled:
            return {
                "state_changed": False,
                "tools_active": False,
                "queries_remaining": 0
            }
                
        if used_tool:
            return {
                "state_changed": False,
                "tools_active": True,
                "queries_remaining": self.tool_disable_threshold
            }
                
        self.consecutive_non_tool_queries += 1
        print(f"Non-tool query counter increased to {self.consecutive_non_tool_queries}")
            
        if self.consecutive_non_tool_queries >= self.tool_disable_threshold:
            result = self.disable_tools(reason="auto")
            return {
                "state_changed": True, 
                "tools_active": False,
                "queries_remaining": 0,
                "message": result["message"]
            }
                
        queries_remaining = self.tool_disable_threshold - self.consecutive_non_tool_queries
        return {
            "state_changed": False, 
            "tools_active": True,
            "queries_remaining": queries_remaining
        }

    def tools_are_active(self) -> bool:
        return self.tools_enabled

    def handle_tool_command(self, query: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Handle tool enabling/disabling commands from voice input.
        
        Args:
            query: The user's query text
            
        Returns:
            Tuple containing:
                - bool: Whether this was a tool command
                - Union[Dict, str]: Either the complete result object (with audio_key) or error message
        """
        try:
            query_lower = query.lower()
            
            has_enable_command = any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES)
            has_disable_command = any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES)
            
            if has_enable_command and has_disable_command:
                print(f"Mixed tool commands detected in: {query_lower}")
                
                # Determine which command appeared last in the query
                last_enable_pos = max((query_lower.rfind(phrase) for phrase in self.TOOL_ENABLE_PHRASES), default=-1)
                last_disable_pos = max((query_lower.rfind(phrase) for phrase in self.TOOL_DISABLE_PHRASES), default=-1)
                
                # Execute the command that appeared last
                if last_disable_pos > last_enable_pos:
                    result = self._handle_disable_command()
                else:
                    result = self._handle_enable_command()
                    
                # Return the full result object, not just the message
                return True, result
            
            elif has_disable_command:
                result = self._handle_disable_command()
                return True, result
            elif has_enable_command:
                result = self._handle_enable_command()
                return True, result
                
            return False, None
                
        except Exception as e:
            print(f"Error in handle_tool_command: {e}")
            # Return a structured error for consistency
            return False, {
                "success": False,
                "message": str(e),
                "state": "error"
            }
        
    def _handle_enable_command(self) -> Dict[str, Any]:
        """Handle a command to enable tools."""
        try:
            if not self.tools_enabled:
                result = self.enable_tools()
                return {
                    "success": True,
                    "message": result["message"],
                    "state": "enabled",
                    "audio_key": "enable"  # Add audio key
                }
            return {
                "success": True,
                "message": "Tools are already enabled.",
                "state": "enabled",
                "audio_key": "already_enabled"  # Add audio key
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error enabling tools: {str(e)}",
                "state": "error"
            }

    def _handle_disable_command(self) -> Dict[str, Any]:
        """Handle a command to disable tools."""
        try:
            if self.tools_enabled:
                result = self.disable_tools(reason="manual")
                if not isinstance(result, dict) or "message" not in result:
                    raise ValueError("Invalid result from disable_tools")
                return {
                    "success": True,
                    "message": result["message"],
                    "state": "disabled",
                    "audio_key": "disable"  # Add audio key
                }
            return {
                "success": True,
                "message": "Tools are already disabled.",
                "state": "disabled",
                "audio_key": "already_disabled"  # Add audio key
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error disabling tools: {str(e)}",
                "state": "error"
            }

    def get_tools_for_query(self, query: str) -> Tuple[bool, List[dict]]:
        """
        Determine if tools are needed and which ones are relevant for a query.
        
        Args:
            query: User's voice query text
            
        Returns:
            Tuple containing:
                - bool: Whether any tools are needed for this query
                - List[dict]: List of relevant tool definitions
        """
        query_lower = query.lower()
        relevant_tools = []
        
        # Check if this is ONLY a tool command
        is_enable = any(phrase in query_lower for phrase in self.TOOL_ENABLE_PHRASES)
        is_disable = any(phrase in query_lower for phrase in self.TOOL_DISABLE_PHRASES)
        
        if is_enable or is_disable:
            # Remove the tool command part from query for further analysis
            for phrase in self.TOOL_ENABLE_PHRASES + self.TOOL_DISABLE_PHRASES:
                query_lower = query_lower.replace(phrase, '').strip()
                
            # If nothing left after removing tool command, return early
            if not query_lower:
                return False, []
        
        # Continue with tool analysis on remaining query
        for category, keywords in self.TOOL_CATEGORY_KEYWORDS.items():
            if any(word in query_lower for word in keywords):
                category_tools = get_tools_by_category(category)
                relevant_tools.extend(category_tools)
        
        # Remove duplicate tools while preserving order
        relevant_tools = list({tool['name']: tool for tool in relevant_tools}.values())
        
        # Debug logging
        print(f"\nTool Analysis:")
        print(f"Query: {query_lower[:50]}...")
        print(f"Contains Tool Command: {is_enable or is_disable}")
        print(f"Tools Found: {[tool['name'] for tool in relevant_tools]}")
        
        return bool(relevant_tools), relevant_tools
    
    def process_confirmation(self, response: str) -> Tuple[bool, bool, str]:
        """
        Process user's response to a confirmation prompt about using tools.
        
        Args:
            response: User's response text
            
        Returns:
            Tuple of (was_confirmation, is_affirmative, message)
        """
        response_lower = response.lower()
        
        # Check if this is a confirmation response
        confirmation_words = {'yes', 'yeah', 'correct', 'right', 'sure', 'okay', 'yep', 'yup'}
        rejection_words = {'no', 'nope', 'don\'t', 'do not', 'negative', 'cancel', 'stop'}
        
        is_confirmation = any(word in response_lower for word in confirmation_words) or \
                         any(word in response_lower for word in rejection_words)
        
        if not is_confirmation:
            return False, False, ""
            
        is_affirmative = any(word in response_lower for word in confirmation_words)
        
        # Update tool state based on response
        if is_affirmative:
            self.enable_tools()
            message = "Tools enabled. I'll use them to assist you."
        else:
            self.disable_tools(reason="declined")
            message = "I'll proceed without using tools."
            
        return True, is_affirmative, message



