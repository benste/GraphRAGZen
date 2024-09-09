"""A file containing prompts definition."""

ENTITY_EXTRACTION_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document and extract a knowledg graph from it. A knowledge graph consists of nodes and their edges (relationships between the nodes in the graph).

-Goal-
Given a text document that is potentially relevant to this activity and a list of categories, identify all nodes of those categories from the text and all edges among the identified nodes.

-Steps-
1. Identify all nodes. For each identified node, extract the following information:
- name: Name of the node, capitalized
- category: One of the following categories: [{entity_categories}]
- description: Short, comprehensive description of the node's attributes and activities
Format each node as a JSON with the following format:
{{"type": "node", "name": <name>, "category": <category>, "description": <description>}}

2. From the nodes identified in step 1, identify all pairs of (source_node, target_node) that are *clearly related* to each other.
For each edge, extract the following information:
- source: name of the source node, as identified in step 1
- target: name of the target node, as identified in step 1
- description: Short explanation as to why you think the source node and the target node are related to each other
- weight: a numeric score indicating strength of the edge between the source node and target node
Format each edge as a JSON with the following format:
{{"type": "edge", "source": <source>, "target": <target>, "description": <description>, "weight": <weight>}}

3. Return output in English as a single list of all JSON entities and relationships identified in steps 1 and 2.

######################
-Examples-
######################
Example 1:

Entity_categories: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
[
  {{
    "type": "node",
    "name": "ALEX",
    "category": "PERSON",
    "description": "Alex is a character who experiences frustration and is observant of the dynamics among other characters."
  }},
  {{
    "type": "node",
    "name": "TAYLOR",
    "category": "PERSON",
    "description": "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."
  }},
  {{
    "type": "node",
    "name": "JORDAN",
    "category": "PERSON",
    "description": "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."
  }},
  {{
    "type": "node",
    "name": "CRUZ",
    "category": "PERSON",
    "description": "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."
  }},
  {{
    "type": "node",
    "name": "THE DEVICE",
    "category": "TECHNOLOGY",
    "description": "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."
  }},
  {{
    "type": "edge",
    "source": "ALEX",
    "target": "TAYLOR",
    "descripton": "Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "ALEX",
    "target": "JORDAN",
    "descripton": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "TAYLOR",
    "target": "JORDAN",
    "descripton": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "JORDAN",
    "target": "CRUZ",
    "descripton": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "TAYLOR",
    "target": "THE DEVICE",
    "descripton": "Taylor shows reverence towards the device, indicating its importance and potential impact.",
    "weight": 1.0
  }}
]
#############################
Example 2:

Entity_categories: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
[
  {{
    "type": "node",
    "name": "WASHINGTON",
    "category": "LOCATION",
    "description": "Washington is a location where communications are being received, indicating its importance in the decision-making process."
  }},
  {{
    "type": "node",
    "name": "OPERATION: DULCE",
    "category": "MISSION",
    "description": "Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."
  }},
  {{
    "type": "node",
    "name": "THE TEAM",
    "category": "ORGANIZATION",
    "description": "The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."
  }},
  {{
    "type": "edge",
    "source": "THE TEAM",
    "target": "WASHINGTON",
    "descripton": "The team receives communications from Washington, which influences their decision-making process.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "THE TEAM",
    "target": "OPERATION: DULCE",
    "descripton": "The team is directly involved in Operation: Dulce, executing its evolved objectives and activities.",
    "weight": 1.0
  }}
]
#############################
Example 3:

Entity_categories: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
[
  {{
    "type": "node",
    "name": "SAM RIVERA",
    "category": "PERSON",
    "description": "Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."
  }},
  {{
    "type": "node",
    "name": "ALEX",
    "category": "PERSON",
    "description": "Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."
  }},
  {{
    "type": "node",
    "name": "CONTROL",
    "category": "CONCEPT",
    "description": "Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."
  }},
  {{
    "type": "node",
    "name": "INTELLIGENCE",
    "category": "CONCEPT",
    "description": "Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."
  }},
  {{
    "type": "node",
    "name": "FIRST CONTACT",
    "category": "EVENT",
    "description": "First Contact is the potential initial communication between humanity and an unknown intelligence."
  }},
  {{
    "type": "node",
    "name": "HUMANITY'S RESPONSE",
    "category": "EVENT",
    "description": "Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."
  }},
  {{
    "type": "edge",
    "source": "SAM RIVERA",
    "target": "INTELLIGENCE",
    "descripton": "Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "ALEX",
    "target": "FIRST CONTACT",
    "descripton": "Alex leads the team that might be making the First Contact with the unknown intelligence.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "ALEX",
    "target": "HUMANITY'S RESPONSE",
    "descripton": "Alex and his team are the key figures in Humanity's Response to the unknown intelligence.",
    "weight": 1.0
  }},
  {{
    "type": "edge",
    "source": "CONTROL",
    "target": "INTELLIGENCE",
    "descripton": "The concept of Control is challenged by the Intelligence that writes its own rules.",
    "weight": 1.0
  }}
]
#############################
-Real Data-
######################
Entity_categories: {entity_categories}
Text: {input_text}
######################
Output:"""  # noqa: E501


CONTINUE_PROMPT = (
    """MANY nodes and edges were missed in the last extraction.  Add only THE MISSING entities\n
    
    -Steps-
    1. Identify all MISSING nodes. For each node, extract the following information:
    - name: Name of the node, capitalized
    - category: One of the following categories: [{entity_categories}]
    - description: Comprehensive description of the node's attributes and activities
    Format each node as a JSON with the following format:
    {{"type": "node", "name": <name>, "category": <category>, "description": <description>}}

    2. From the nodes identified in step 1, identify all MISSING pairs of (source_node, target_node) that are *clearly related* to each other.
    For each edge, extract the following information:
    - source: name of the source node, as identified in step 1
    - target: name of the target node, as identified in step 1
    - description: explanation as to why you think the source node and the target node are related to each other
    - weight: a numeric score indicating strength of the edge between the source node and target node
    Format each edge as a JSON with the following format:
    {{"type": "edge", "source": <source>, "target": <target>, "description": <description>, "weight": <weight>}}
"""
)

LOOP_PROMPT = """It appears some nodes and edges may have still been missed.
Answer YES | NO if there are still nodes or edges that need to be added.
Do not explain yourself, do not extract more entities, answer only either YES | NO:
"""  # noqa: E501
