from flask import Flask, request, render_template, jsonify
from VZsampleEnv import MobilePhoneCarrierEnv, Actions
from stable_baselines3 import PPO


env = MobilePhoneCarrierEnv()
obs,_ = env.reset()
model=PPO.load("activation")
action, _state = model.predict(obs, deterministic=False)
#action = env.action_space.sample()
print("Doing ", Actions.get_action_type(action))
obs, reward, done, terminated, info = env.step(action)

app = Flask(__name__)
params = ""
@app.route("/", methods=["GET"])
def index():
    new_state = env.observation_space.sample()
    return render_template("index.html", state=new_state,action=0,completed=0)

@app.route("/predict", methods=["POST"])

def predict():
    done=False
    new_state = env.observation_space.sample()
    new_state["address_validation_status"] = int(request.form.get('address_validation_status'))
    new_state["device_validation_status"] = int(request.form.get('device_validation_status'))
    new_state["sim_validation_status"] = int(request.form.get('sim_validation_status'))
    new_state["new_mdn_status"] = int(request.form.get('new_mdn_status'))
    new_state["existing_mdn_status"] = int(request.form.get('existing_mdn_status'))
    new_state["payment_status"] = int(request.form.get('payment_status'))
    new_state["use_existing_mdn"] = int(request.form.get('use_existing_mdn'))
    new_state["vas_status"] = int(request.form.get('vas_status'))
    
    # Run the model inference
    action, _state = model.predict(new_state)
    obs, reward, done, terminated, info = env.step(action)
    # Format the prediction for display

    if obs is not None:
        state_dict = dict(obs)
    else:
        state_dict = {}  # Or a placeholder value

    action_taken = Actions.get_action_type(action)
    if done:
        completed=1
        env.reset();
        state_dict=env.observation_space.sample();
    else:
        completed=0
    #return jsonify({"result": result})
    return render_template("index.html",state=state_dict, action=action_taken, completed=completed)

if __name__ == "__main__":
    app.run(debug=True)