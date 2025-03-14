import streamlit as st
import pandas as pd
import joblib

# Custom CSS for vibrant yellow UI + Pokémon trainers
st.markdown("""
    <style>
    .stApp {
        background-color: #FFD95F !important;
    }
    .appview-container, .main {
        background-color: #FFD95F !important;
    }
    .block-container {
        background-color: #FFD95F6 !important;
        padding: 2rem 1rem;
    }
    .title {
        font-size: 48px;
        color: #d32f2f;
        text-align: center;
    }
    .subtitle {
        font-size: 22px;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextInput > label {
        font-size: 18px;
        color: #212121;
    }
    .stButton > button {
        background-color: #ffca28;
        color: black;
        border-radius: 10px;
        font-size: 18px;
    }
    .pokemon-box {
        background-color: #fffde7;
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed #f57f17;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("pokemon_model.pkl")
    encoder = joblib.load("encoder.pkl")
    pokemon_classes = joblib.load("pokemon_classes.pkl")
    return model, encoder, pokemon_classes

def predict_pokemon(ability1, ability2, hidden_ability, model, encoder, pokemon_classes):
    # Replace blank strings with "none" and convert to lowercase
    ability1 = ability1.strip().lower() if ability1.strip() != "" else "none"
    ability2 = ability2.strip().lower() if ability2.strip() != "" else "none"
    hidden_ability = hidden_ability.strip().lower() if hidden_ability.strip() != "" else "none"

    input_data = pd.DataFrame([[ability1, ability2, hidden_ability]], columns=['Ability1', 'Ability2', 'HiddenAbility'])
    encoded_input = encoder.transform(input_data)
    predicted_class = model.predict(encoded_input)[0]
    return pokemon_classes[predicted_class]

# Load model
model, encoder, pokemon_classes = load_model()

# Top Pokémon human character banner
st.markdown("<br><br>", unsafe_allow_html=True)  # Add 2 line breaks
st.image("https://apim.partycity.ca/v1/product/api/v1/product/image/8436444p?baseStoreId=PTY&lang=en_CA&subscription-key=c01ef3612328420c9f5cd9277e815a0e&imwidth=640&impolicy=mZoom", caption="Ash, Misty & Brock", use_container_width=True)

# Title and subtitle
st.markdown('<div class="title">✨ Pokémon Predictor 🔮</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the abilities and discover which Pokémon it could be! 🧠⚡</div>', unsafe_allow_html=True)

# Input fields
st.markdown("### 🔢 Enter Pokémon Abilities:")
col1, col2, col3 = st.columns(3)
with col1:
    ability1 = st.text_input("🛡️ Ability 1").strip()
with col2:
    ability2 = st.text_input("✨ Ability 2 (optional)", value="None").strip()
with col3:
    hidden_ability = st.text_input("🔮 Hidden Ability (optional)", value="None").strip()

# Predict Button
if st.button("🔍 Predict Pokémon"):
    if ability1:
        try:
            predicted_pokemon = predict_pokemon(ability1, ability2, hidden_ability, model, encoder, pokemon_classes)
            st.markdown(f'<div class="pokemon-box">🎉 Predicted Pokémon: <strong>{predicted_pokemon.title()}</strong> ⚡</div>', unsafe_allow_html=True)
            image_url = f"https://img.pokemondb.net/artwork/large/{predicted_pokemon.lower().replace(' ', '-')}.jpg"
            st.image(image_url, caption=f"{predicted_pokemon.title()}", use_container_width=True)
        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter at least Ability 1.")

# Suggestion Box
with st.expander("💡 Try Sample Abilities"):
    st.markdown("Try some of these common ability combos:")
    st.code("""
Ability 1: Overgrow
Ability 2: Chlorophyll
Hidden Ability: Leaf Guard
""")

# Bottom Pokémon Trainer Image
st.image("https://images.alphacoders.com/473/thumb-1920-473848.png", caption="Pikachu!", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Gotta Predict 'Em All! 🔥")
