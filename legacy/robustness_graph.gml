graph [
  directed 1
  node [
    id 0
    label "System Interface"
    type "boundary"
    shared 0
    color "#9b59b6"
    shape "circle"
  ]
  node [
    id 1
    label "Notification Interface"
    type "boundary"
    shared 0
    color "#9b59b6"
    shape "circle"
  ]
  node [
    id 2
    label "Configuration Interface"
    type "boundary"
    shared 0
    color "#9b59b6"
    shape "circle"
  ]
  node [
    id 3
    label "Espressos"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 4
    label "Pressure"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 5
    label "beans"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 6
    label "Milk"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 7
    label "Container"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 8
    label "Level"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 9
    label "Temperature"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 10
    label "Heater"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 11
    label "Time"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 12
    label "Grounds"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 13
    label "Notification"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 14
    label "Coffee"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 15
    label "Amount"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 16
    label "Sugar"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 17
    label "Compressor"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 18
    label "Water"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 19
    label "User"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 20
    label "Filter"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 21
    label "Cup"
    type "entity"
    shared 0
    color "#27ae60"
    shape "square"
  ]
  node [
    id 22
    label "Amount Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 23
    label "Coffee Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 24
    label "Error Handler A2"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 25
    label "System Orchestrator"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 26
    label "Milk Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 27
    label "Filter Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 28
    label "Error Handler A1"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 29
    label "Heater Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 30
    label "Cup Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 31
    label "Notification Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 32
    label "Compressor Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 33
    label "Begin Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 34
    label "Time Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  node [
    id 35
    label "Water Controller"
    type "control"
    shared 0
    color "#3498db"
    shape "triangle"
  ]
  edge [
    source 19
    target 0
    type "interacts"
    uc "UC2: Prepare Espresso"
    step "B8"
  ]
  edge [
    source 19
    target 1
    type "interacts"
    uc "UC2: Prepare Espresso"
    step "B8"
  ]
  edge [
    source 25
    target 34
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B1"
  ]
  edge [
    source 25
    target 29
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B2"
  ]
  edge [
    source 25
    target 27
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B5"
  ]
  edge [
    source 25
    target 22
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B4"
  ]
  edge [
    source 25
    target 30
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B5"
  ]
  edge [
    source 25
    target 35
    type "coordinates"
    uc "UC1: Prepare Latte"
    step "B7"
  ]
  edge [
    source 25
    target 33
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B7"
  ]
  edge [
    source 25
    target 26
    type "coordinates"
    uc "UC1: Prepare Latte"
    step "B8"
  ]
  edge [
    source 25
    target 23
    type "coordinates"
    uc "UC1: Prepare Latte"
    step "B9"
  ]
  edge [
    source 25
    target 31
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B8"
  ]
  edge [
    source 25
    target 32
    type "coordinates"
    uc "UC2: Prepare Espresso"
    step "B6"
  ]
]
