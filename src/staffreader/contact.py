"""Who this tool's web requests say they come from.

IMSLP, the Internet Archive, Wikimedia and the MDZ ask automated tools to
name themselves and give a contact in their User-Agent, and the Archive
asks an AI agent to name its model too. Those are the user's details, so
they never live in the code: the contact and agent settings give them
(settings.py), and an agent's model is left out when a person runs the
tool. A client refuses to fetch without a contact rather than send an
anonymous request."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import settings

TOOL = 'staffreader'


class NoContact(RuntimeError):
    pass


def _version() -> str:
    try:
        return version('staffreader')
    except PackageNotFoundError:
        return '0'


def user_agent(short: bool = False) -> str:
    """The User-Agent: the tool, its version, the contact and the agent's
    model; [short] gives the tool and contact alone, as IMSLP is sent."""
    contact = settings.get('contact')
    if not contact:
        raise NoContact(f'no contact for web requests: set contact in {settings.config_file()} '
                        'or STAFFREADER_CONTACT (see settings.py)')
    if short:
        return f'{TOOL} ({contact})'
    agent = settings.get('agent')
    return f'{TOOL}/{_version()} ({contact})' + (f' ({agent})' if agent else '')
