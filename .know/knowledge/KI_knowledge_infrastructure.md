<!-- last_verified: 2026-05-13 -->
# KI: Knowledge Infrastructure (Техническая реализация)

## Overview
Техническое ядро системы управления знаниями: сбор метрик, отслеживание хешей и предоставление MCP-интерфейса.

## Key Components

### Core Engine
| Class / Function | File | Purpose |
|---|---|---|
| `KnowledgeEngine` | `scripts/knowledge_engine.py` | Расчет SHA-256 хешей, обнаружение изменений и маппинг. |
| `KnowledgeManager` | `scripts/knowledge_mcp.py` | Реализация MCP-интерфейса для инструментов `audit_coverage`, `save_state` и др. |

### Tooling
| Script | Purpose | Details |
|---|---|---|
| `audit_coverage.py` | Анализ качества. | Расчет Density, Complexity и формирование матрицы покрытия. |
| `generate_dir_index.py` | Индексация папок. | Генерация `DIR_INDEX.md` на основе сканирования файловой системы. |
| `sync_agents_md.py` | Синхронизация инструкций. | Обновление таблицы KI в `AGENTS.md`. |

## Technical Details
- **State Tracking**: `doc_state.json` хранит хеши всех отслеживаемых файлов. Это позволяет мгновенно определять, какие KI устарели (Stale).
- **Jailhouse Security**: Все операции через MCP ограничены директорией `.know/` для предотвращения случайного повреждения исходного кода проекта.

## Non-obvious Details
- **Dependency Inversion**: KI не знают о коде напрямую; связь устанавливается в `doc_config.json`, что позволяет менять структуру документации без правки исходников.
- **Auto-verification**: При каждом сохранении (`save_state`) система проверяет целостность манифеста.

## Common Pitfalls
- **Mtime Desync**: Некоторые редакторы не обновляют время модификации файла при сохранении идентичного контента, что может обмануть оптимизацию движка. Используйте `save_state` принудительно.



## Related KIs

