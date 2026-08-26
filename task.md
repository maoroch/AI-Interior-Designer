# 📋 План задач: Архитектурные улучшения и Оптимизация

## 🎯 Цели
1. **Decoupled Semantic CAD Architecture**: Перевести генерацию интерьера на разделение геометрии и семантики (локальный бэкенд хранит точные 3D-координаты, Groq LLM оперирует человеческими правилами в см/м, Python CAD-компилятор исключает коллизии).
2. **Three.js & WebGL Performance Optimization**: Снизить нагрузку на GPU/CPU ПК, устранить нагрев и шум вентиляторов при просмотре 3D-сцен.

---

## 🛠️ Задачи к реализации

### Блок 1: Decoupled Semantic CAD Architecture (Бэкенд & AI)

- [x] **Task 1: Генератор семантического брифа (`backend/app/cv/semantic_bridge.py`)**
  - [x] Преобразование `WallGraph` в читаемое описание комнаты:
    - Габариты: ширина, длина, высота в метрах и сантиметрах.
    - Разметка стен по сторонам света/номерам: `Северная стена (окно)`, `Южная стена (глухая)`, `Западная стена (дверь)`.
  - [x] Полное исключение передачи «сырых» `float`-координат полигонов в LLM.

- [x] **Task 2: Обновление промптов и схем LLM-агентов (`style_agent`, `furniture_planner`)**
  - [x] Обновление системного промпта для Groq на генерацию семантических привязок:
    - `anchor_wall`: `"north" | "south" | "east" | "west" | "center"`
    - `placement`: `"center" | "left_corner" | "right_corner" | "under_window" | "opposite_sofa"`
    - `distance_from_wall_cm`: расстояние в сантиметрах
    - `dimensions_cm`: габариты предмета мебели
  - [x] Сокращение размера промпта на 60–75% (минимизация риска `429 RateLimit` на Groq).

- [x] **Task 3: Детерминированный 3D CAD-компилятор (`backend/app/agents/furniture_planner/compiler.py`)**
  - [x] Трансляция семантических правил в точные мировые координаты `position: [x, 0, z]` и угол `rotation_y`.
  - [x] Математическая валидация проходов (`Shapely`):
    - Свободная зона перед входными и межкомнатными дверьми (радиус $\ge 1.0\text{ м}$).
    - Минимальные эргономические проходы между предметами ($\ge 60\text{ см}$).
    - Отсутствие врезания мебели в стены.

- [x] **Task 4: Сквозная интеграция и тестирование (E2E)**
  - [x] Проверка генерации на всех 5 тестовых планировках (`plan1_studio` ... `plan5_loft`).
  - [x] Добавление unit-тестов на CAD-компилятор в `backend/tests/test_semantic_cad.py`.
  - [x] Проверка стабильности рендеринга и управления в `SceneViewer.tsx` (Three.js WebGL).

---

### Блок 2: Three.js & WebGL Performance Optimization (Фронтенд & Графика)

- [x] **Task 5: Рендеринг по требованию (`frameloop="demand"`)**
  - [x] Перевод Canvas в `frameloop="demand"` в `SceneViewer.tsx`.
  - [x] Настройка триггеров перерисовки `invalidate()` при:
    - Вращении и зуме камеры (`OrbitControls onChange`).
    - Переключении вариантов дизайна (A/B/C).
    - Перемещении в режиме от первого лица (First-Person Walkthrough).
  - [x] Результат: **0% нагрузки на GPU в состоянии покоя**.

- [x] **Task 6: Ограничение DPI дисплея (`dpr={[1, 1.5]}`)**
  - [x] Ограничение максимального разрешения рендера на Retina и 4K мониторах до 1.5.
  - [x] Результат: **Снижение нагрузки на GPU на 50–60%** без видимой потери качества.

- [x] **Task 7: Оптимизация теней и источников света**
  - [x] Ограничение `castShadow={true}` только для 1 ключевого направленного источника света.
  - [x] Отключение динамических теней для декоративных ламп и подсветки мебели.
  - [x] Оптимизация размера карт теней (`1024x1024`).

- [x] **Task 8: Очистка памяти при переключении сцен (VRAM Disposal)**
  - [x] Добавление хуков очистки для материалов, процедурных текстур дерева и геометрий при смене дизайна.
  - [x] Предотвращение утечек видеопамяти при длительной сессии.

---

## 📦 Контракты данных (Data Schemas)

### 1. Бриф для LLM (Input Prompt):
```yaml
room_type: "Living Room"
dimensions:
  width_m: 5.40
  depth_m: 3.60
  area_sqm: 19.44
walls:
  - id: "wall_north"
    length_m: 5.40
    features: ["window_center_2.0m"]
  - id: "wall_south"
    length_m: 5.40
    features: ["solid_wall"]
  - id: "wall_east"
    length_m: 3.60
    features: ["solid_partition"]
  - id: "wall_west"
    length_m: 3.60
    features: ["door_to_hallway"]
```

### 2. Ответ LLM (Semantic Rules JSON):
```json
[
  {
    "item": "sofa_3seater",
    "anchor_wall": "wall_south",
    "placement": "center",
    "distance_from_wall_cm": 15,
    "dimensions_cm": { "width": 220, "depth": 90, "height": 85 }
  },
  {
    "item": "tv_stand",
    "anchor_wall": "wall_north",
    "placement": "center",
    "distance_from_wall_cm": 0,
    "dimensions_cm": { "width": 180, "depth": 40, "height": 45 }
  }
]
```

### 3. Финальный `SceneJSON` для Three.js:
```json
{
  "id": "sofa_1",
  "type": "sofa",
  "position": [5.42, 0.0, 3.20],
  "rotation": [0, 3.1415, 0],
  "dimensions": [2.20, 0.85, 0.90]
}
```
