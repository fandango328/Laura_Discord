"""
LAURA Tools Configuration
Last Updated: 2024-03-13

Purpose:
    Defines available tools and their schemas for LAURA voice assistant.
    Structured for compatibility with various LLM APIs.
"""

# Tool definitions
AVAILABLE_TOOLS = [
    {
        "name": "draft_email",
        "description": "Draft a new email with a recipient and content. This tool creates a draft email in the user's Gmail account. It should be used when the user wants to compose or prepare an email message. The tool will create the draft but will not send it automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Email address of who will receive the email"
                },
                "subject": {
                    "type": "string",
                    "description": "The topic or subject line of the email"
                },
                "content": {
                    "type": "string",
                    "description": "The main body content of the email"
                }
            },
            "required": ["subject", "content"]
        }
    },
    {
        "name": "calendar_query",
        "description": "Get information about calendar events. Can retrieve next event or full day schedule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["next_event", "day_schedule", "check_availability"],
                    "description": "Type of calendar query ('next_event' for next upcoming event, 'day_schedule' for full day's schedule)"
                },
                "date": {
                    "type": "string",
                    "description": "Date for query (optional)"
                }
            },
            "required": ["query_type"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current time in the local timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["time", "date", "both"],
                    "description": "Format of time information to return"
                }
            },
            "required": ["format"]
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Event title"
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format"
                },
                "description": {
                    "type": "string",
                    "description": "Event description"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses"
                }
            },
            "required": ["summary", "start_time", "end_time"]
        }
    },
    {
        "name": "get_location",
        "description": "Get current location based on WiFi networks. This tool scans nearby WiFi networks and uses them to determine the current geographic location. It can return the location in different formats including coordinates or street address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["coordinates", "address", "both"],
                    "description": "Format of location information to return"
                }
            },
            "required": ["format"]
        }
    },
    {
        "name": "calibrate_voice_detection",
        "description": "Calibrate the voice detection system to improve speech recognition. This tool runs a calibration process that measures background noise and voice levels to optimize how the system detects when you're speaking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Confirmation to run calibration"
                }
            },
            "required": ["confirm"]
        }
    },
    {
        "name": "read_emails",
        "description": "Retrieve and read emails from the user's Gmail inbox with various filtering options. Can identify important emails based on sender, content, and urgency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to retrieve (default: 5)"
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "Whether to only return unread emails"
                },
                "query": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'from:john@example.com', 'subject:meeting', 'after:2023/04/01')"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to include the full email content or just headers"
                },
                "mark_as_read": {
                    "type": "boolean", 
                    "description": "Whether to mark retrieved emails as read"
                }
            }
        }
    },
    {
        "name": "email_action",
        "description": "Perform actions on specific emails like archive, delete, or mark as read/unread",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "ID of the email to act upon"
                },
                "action": {
                    "type": "string",
                    "enum": ["archive", "trash", "mark_read", "mark_unread", "star", "unstar"],
                    "description": "Action to perform on the email"
                }
            },
            "required": ["email_id", "action"]
        }
    },
    {
        "name": "update_calendar_event",
        "description": "Update an existing calendar event's details such as time, location, or attendees",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to update"
                },
                "summary": {
                    "type": "string",
                    "description": "New title for the event (optional)"
                },
                "start_time": {
                    "type": "string",
                    "description": "New start time in ISO format (optional)"
                },
                "end_time": {
                    "type": "string",
                    "description": "New end time in ISO format (optional)"
                },
                "description": {
                    "type": "string",
                    "description": "New description for the event (optional)"
                },
                "location": {
                    "type": "string",
                    "description": "New location for the event (optional)"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated list of attendee email addresses (optional)"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "cancel_calendar_event",
        "description": "Cancel an existing calendar event and optionally notify attendees",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to cancel"
                },
                "notify_attendees": {
                    "type": "boolean",
                    "description": "Whether to send cancellation emails to attendees"
                },
                "cancellation_message": {
                    "type": "string",
                    "description": "Optional message to include in the cancellation notification"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "manage_tasks",
        "description": "Create, update, list, or complete tasks in Google Tasks",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "list", "complete", "delete"],
                    "description": "Action to perform on tasks"
                },
                "title": {
                    "type": "string",
                    "description": "Title of the task (for create/update)"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes or details for the task"
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date for the task in ISO format or natural language (e.g., 'tomorrow')"
                },
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to update/complete/delete"
                },
                "status": {
                    "type": "string",
                    "enum": ["needsAction", "completed"],
                    "description": "Status of the task"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of tasks to return when listing"
                },
                "list_id": {
                    "type": "string",
                    "description": "ID of the task list (optional, uses default if not specified)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "create_task_from_email",
        "description": "Create a task based on an email",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "ID of the email to convert to a task"
                },
                "title": {
                    "type": "string",
                    "description": "Custom title for the task (optional, will use email subject if not provided)"
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date for the task (optional)"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Priority level for the task"
                }
            },
            "required": ["email_id"]
        }
    },
    {
        "name": "create_task_for_event",
        "description": "Create preparation or follow-up tasks for a calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "ID of the calendar event to create tasks for"
                },
                "task_type": {
                    "type": "string",
                    "enum": ["preparation", "follow_up", "both"],
                    "description": "Type of tasks to create"
                },
                "days_before": {
                    "type": "integer",
                    "description": "Days before the event for preparation tasks"
                },
                "days_after": {
                    "type": "integer",
                    "description": "Days after the event for follow-up tasks"
                },
                "custom_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom titles for tasks (optional)"
                }
            },
            "required": ["event_id", "task_type"]
        }
    },
    {
        "name": "manage_contacts",
        "description": "Manage contact information for people and organizations",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "create", "update", "list", "search"],
                    "description": "Action to perform on contacts"
                },
                "name": {
                    "type": "string",
                    "description": "Full name of the contact"
                },
                "email": {
                    "type": "string",
                    "description": "Email address of the contact"
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number of the contact"
                },
                "company": {
                    "type": "string",
                    "description": "Organization or company name"
                },
                "relationship": {
                    "type": "string",
                    "description": "Relationship to the user (e.g., 'manager', 'colleague', 'client')"
                },
                "importance": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Importance level of the contact"
                },
                "query": {
                    "type": "string",
                    "description": "Search term for finding contacts"
                }
            },
            "required": ["action"]
        }
    }
]

# Tool categories for easier reference
EMAIL_TOOLS = ['draft_email', 'read_emails', 'email_action']
CALENDAR_TOOLS = ['calendar_query', 'create_calendar_event', 'update_calendar_event', 'cancel_calendar_event']
TASK_TOOLS = ['manage_tasks', 'create_task_from_email', 'create_task_for_event']
UTILITY_TOOLS = ['get_current_time', 'get_location', 'calibrate_voice_detection']
CONTACT_TOOLS = ['manage_contacts']

def get_tool_by_name(tool_name):
    """Get tool definition by name."""
    return next((tool for tool in AVAILABLE_TOOLS if tool['name'] == tool_name), None)

def get_tools_by_category(category):
    """Get all tools in a category."""
    return [tool for tool in AVAILABLE_TOOLS if tool['name'] in globals().get(f"{category}_TOOLS", [])]