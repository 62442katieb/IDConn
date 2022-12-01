{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

    {% block classes %}
        {% if classes %}
            .. rubric:: Classes

            .. autosummary::
                :toctree:
                :template: class.rst
                {% for item in classes %}
                    {{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block functions %}
        {% if functions %}
            .. rubric:: Functions

            .. autosummary::
              :toctree:
              :template: function.rst
              {% for item in functions %}
                  {{ item }}
              {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block exceptions %}
        {% if exceptions %}
            .. rubric:: Exceptions

            .. autosummary::
              :toctree:
              {% for item in exceptions %}
                  {{ item }}
              {%- endfor %}
        {% endif %}
    {% endblock %}
