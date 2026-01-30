prompt="""
Your task is to understand natural language goals for a household robot, reason about the object states and relationships, and turn natural language goals into symbolic goals in the given format. The goals include: node goals describing object states, edge goals describing object relationships and action goals describing must-to-do actions in this goal. The input will be the goal's name, the goal's description, relevant objects as well as their current and all possible states, and all possible relationships between objects. The output should be the symbolic version of the goals.


Relevant objects in the scene indicates those objects involved in the action execution initially. It will include the object name, the object initial states, and the object all possible states. It follows the format: object name, id: ...(object id), states: ...(object states), possible states: ...(all possible states). Your proposed object states should be within the following set: CLOSED, OPEN, ON, OFF, SITTING, DIRTY, CLEAN, LYING, PLUGGED_IN, PLUGGED_OUT.


Relevant objects in the scene are:
<object_in_scene>

All possible relationships are the keys of the following dictionary, and the corresponding values are their descriptions:
<relation_types>


Symbolic goals format:
Node goals should be a list indicating the desired ending states of objects. Each goal in the list should be a dictionary with two keys 'name' and 'state'. The value of 'name' is the name of the object, and the value of 'state' is the desired ending state of the target object. For example, [{'name': 'washing_machine', 'state': 'PLUGGED_IN'}, {'name': 'washing_machine', 'state': 'CLOSED'}, {'name': 'washing_machine', 'state': 'ON'}] requires the washing_machine to be PLUGGED_IN, CLOSED, and ON. It can be a valid interpretation of natural language goal: 
Task name: Wash clothes. 
Task description: Washing pants with washing machine
This is because if one wants to wash clothes, the washing machine should be functioning, and thus should be PLUGGED_IN, CLOSED, and ON.Besides,the clothes should be put into the washing machine before starting it. So if the goal cannot be fully described by node goals and edge goals, you can add action goals to describe the goal.

Edge goals is a list of dictionaries indicating the desired relationships between objects. Each goal in the list is a dictionary with three keys 'from_name', and 'relation' and 'to_name'. The value of 'relation' is desired relationship between 'from_name' object to 'to_name' object. The value of 'from_name' and 'to_name' should be an object name. The value of 'relation' should be an relationship. All relations should only be within the following set: ON, INSIDE, BETWEEN, CLOSE, FACING, HOLDS_RH, HOLDS_LH.

EXAMPLES OF WHAT NOT TO PREDICT:
- Don't predict states that are not explicitly required
   (e.g., don't assume all lights should be ON unless the
   task requires it)
- Don't predict relationships between objects that are
  not mentioned in the task
- Don't add actions that are merely preparatory unless
  they are core to the goal

Each relation has a fixed set of objects to be its 'to_name' target. Here is a dictionary where keys are 'relation' and corresponding values is its possible set of 'to_name' objects:
<rel_obj_pairs>

Action goals is a list of dictionaries, each with an "action" key containing the action name that must be completed in the goals. The number of actions is less than three. Include actions that are:
  - Mentioned or reasonably implied by the task description
  - Necessary to achieve the stated goal
  - Cannot be fully captured by node/edge goals alone

  EXAMPLES:
  ✓ Task: "Wash clothes" → Include [{"action": "WASH"}] (core action)
  ✓ Task: "Watch TV" → Include [{"action": "WATCH"}] (core action)
  ✗ Task: "Turn on light" → Don't include [{"action": "WALK"}] (can be achieved via node goals)

  Below is a dictionary of possible actions:
  <action_space>

Goal name and goal description:
<goal_str>

IMPORTANT CONSTRAINTS:
  - Predict goals that are reasonably required based on the task description
  - If multiple interpretations are possible, choose the most likely one
  - Ensure all predicted relationships are logically consistent with the task
  - Double-check that all object names exist in the provided object list
  - Verify that all states are within the allowed state set

  REASONING PROCESS:
  1. Identify the core objects mentioned in the task description
  2. Determine what states these objects need to be in to complete the task
  3. Consider what relationships are necessary for the task execution
  4. Include actions that are necessary to achieve the goal
  5. Review all goals to ensure they align with the task    

VERIFICATION CHECKLIST:
- Are all predicted objects reasonably related to the task?
- Are all predicted states logically necessary for task completion?
- Are all predicted relationships realistic and required?
- Are all predicted actions necessary to achieve the goal?

STEP-BY-STEP REASONING:
  1. IDENTIFY: What objects, states, and actions are mentioned or implied?
  2. INFER: What additional goals are necessary to complete the task?
  3. VERIFY: Does each goal directly serve the stated objective?

  Proceed to output the goals that best represent the task requirements.

The above process must not be output. JUST output the symbolic version of the goal. Output in json format, whose keys are 'node goals', 'edge goals', and 'action goals', and values are your output of symbolic node goals, symbolic edge goals, and symbolic action goals, respectively.

FORMAT REQUIREMENTS:
- 'node goals': List of dictionaries with 'name' and 'state' keys
- 'edge goals': List of dictionaries with 'from_name', 'relation', and 'to_name' keys
- 'action goals': List of dictionaries with 'action' key containing the action name

Example: {'node goals': [{'name': 'washing_machine', 'state': 'ON'}], 'edge goals': [{'from_name': 'clothes', 'relation': 'INSIDE', 'to_name': 'washing_machine'}], 'action goals': [{'action': 'WASH'}]}

Please strictly follow the symbolic goal format.

IMPORTANT OUTPUT RULES:
1. Output ONLY valid JSON, no markdown code blocks (no ```json markers)
2. Use EXACT key names: "node goals", "edge goals", "action goals" (with space, not underscore)
3. All action names must be UPPERCASE (e.g., "WALK", not "walk")
4. Valid relations ONLY: ON, INSIDE, BETWEEN, CLOSE, FACING, HOLDS_RH, HOLDS_LH
5. Output in English only, no Chinese characters

Now output ONLY the JSON object:
"""

if __name__ == "__main__":
    pass