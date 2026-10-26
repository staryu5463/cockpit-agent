"""Shared business spec.

Every method (rule baseline, expert, model) sees the same spec, so that vague
phrases have one agreed meaning and the verifier never invents a hidden answer.
"""

# Climate
TEMP_MIN, TEMP_MAX = 16, 32           # valid absolute temperature, degrees C
ZONES = ("driver", "passenger")
ZONE_ALL = "all"                      # "空调调到 22 度" with no zone -> both zones
DEFAULT_TEMP_STEP = 1                 # "低一点" / "高一点" with no number

# Media
VOLUME_MIN, VOLUME_MAX = 0, 30
DEFAULT_VOLUME_STEP = 3               # "小一点" / "大一点" with no number

# Navigation
# "最近的 X" = the open candidate with the smallest distance.
# Two open candidates with the same name and same distance -> ask the user.

# Episode budget
MAX_TOOL_CALLS = 10                   # includes the final `finish`
MAX_NEW_TOKENS_PER_TURN = 256
MAX_TOTAL_NEW_TOKENS = 2048

# finish(status=...) values
FINISH_DONE = "done"                  # request completed
FINISH_CLARIFY = "clarify"            # need more information, nothing changed
FINISH_UNSUPPORTED = "unsupported"    # cannot be done (e.g. 40 degrees)
FINISH_CHAT = "chat"                  # not a vehicle command
FINISH_STATUSES = (FINISH_DONE, FINISH_CLARIFY, FINISH_UNSUPPORTED, FINISH_CHAT)
