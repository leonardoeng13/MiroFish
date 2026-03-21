"""
API routing module
==================

Declares the three Flask :class:`~flask.Blueprint` objects used by the
application and imports their route handlers so that all endpoints are
registered before the app starts.

Blueprint layout
----------------
- ``graph_bp``      → ``/api/graph/…``       (project + knowledge-graph routes)
- ``simulation_bp`` → ``/api/simulation/…``  (OASIS simulation routes)
- ``report_bp``     → ``/api/report/…``      (report generation & export routes)

Adding a new blueprint
----------------------
1. Create a new module under ``app/api/``.
2. Import it here alongside the existing blueprints.
3. Register it in :func:`app.create_app` with an appropriate URL prefix.
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
simulation_bp = Blueprint('simulation', __name__)
report_bp = Blueprint('report', __name__)

from . import graph  # noqa: E402, F401
from . import simulation  # noqa: E402, F401
from . import report  # noqa: E402, F401
