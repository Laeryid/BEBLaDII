<!-- last_verified: 2026-05-13 -->
# KI: Knowledge Management (Методология)

## Overview
Философия и стандарты ведения базы знаний проекта. Система базируется на принципе "Документация как код" (Docs as Code) и обеспечивает полную прозрачность архитектурных решений.

## Standards

### Knowledge Items (KI)
Каждый KI должен быть атомарным и содержать:
- **Metadata**: `last_verified`, `covers`, `depends_on`.
- **Context**: Overview, Key Components.
- **Deep Knowledge**: Non-obvious Details, Common Pitfalls.

### Architectural Decision Records (ADR)
Фиксируются в папке `decisions/` и регистрируются в [Architectural Decisions](KI_architectural_decisions.md).

## Quality Matrix
Система оценивает качество по трем осям:
1. **Coverage**: Все ли критические модули имеют связанные KI.
2. **Density**: Достаточно ли деталей в KI относительно объема кода (цель > 50 B/KB).
3. **Freshness**: Соответствует ли содержимое KI текущему состоянию кода.





## Related KIs

