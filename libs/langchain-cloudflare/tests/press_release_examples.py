"""Public press-release inputs shared by chat and decision integration tests."""

# MARK: - EXAMPLES
PARTNERSHIP_SENTENCES = (
    "Acme Corp (NYSE: ACME) today announced a strategic partnership with "
    "TechGiant Inc to jointly develop next-generation AI solutions.",
    "The partnership will combine Acme's expertise in cloud infrastructure "
    "with TechGiant's machine learning capabilities.",
)
PRESS_RELEASE_TEXTS = (
    "\n".join(PARTNERSHIP_SENTENCES),
    "Apple Inc announced record Q4 earnings, beating analyst expectations.",
)

PRESS_RELEASE_EXAMPLES = {
    "press_release_partnership": {
        "text": PRESS_RELEASE_TEXTS[0],
        "sentences": PARTNERSHIP_SENTENCES,
        "announcement": True,
        "type": "partnership",
        "context": "S0",
        "roles": {"Acme Corp": "Partner", "TechGiant Inc": "Partner"},
        "tickers": {"Acme Corp": "ACME", "TechGiant Inc": "none"},
    },
    "press_release_earnings": {
        "text": PRESS_RELEASE_TEXTS[1],
        "sentences": (PRESS_RELEASE_TEXTS[1],),
        "announcement": False,
        "type": "none",
        "context": "none",
        "roles": {"Apple Inc": "None"},
        "tickers": {"Apple Inc": "none"},
    },
}
