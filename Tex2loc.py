from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
from geopy.geocoders import Nominatim
import re

class Tex2loc:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        self.geolocator = Nominatim(user_agent="tex2loc")

        self.reply = ""
        self.city = "Unknown"
        self.country = "Unknown"
        self.continent = "Unknown"

    def generate_location(self, text):
        prompt = f"""
        Based on the following text, identify the most probable location. Try to extract the City and Country.
        If there is not enough information, try to infer at least the Continent based on any clues.
        If you cannot infer anything at all, write "Unknown" for all fields.

        Text: "{text}"

        Answer in the following format exactly:
        City: <city name or "Unknown">
        Country: <country name or "Unknown">
        Continent: <continent name or "Unknown">
        """
        result = self.pipe(prompt, max_new_tokens=100, temperature=0.3)
        self.reply = result[0]['generated_text']
        return self.reply

    def parse_location(self):
        city_match = re.search(r'City: \s*(?!<)(.*)', self.reply)
        country_match = re.search(r'Country: \s*(?!<)(.*)', self.reply)
        continent_match = re.search(r'Continent: \s*(?!<)(.*)', self.reply)

        self.city = city_match.group(1).strip() if city_match else "Unknown"
        self.country = country_match.group(1).strip() if country_match else "Unknown"
        self.continent = continent_match.group(1).strip() if continent_match else "Unknown"

        return self.city, self.country, self.continent

    def get_coordinates(self):
        search_query = ""
        if(self.city != "") :
            search_query = f"{self.city}, {self.country}"
        elif(self.country != "") :
            search_query = f"{self.country}"
        else :
            search_query = f"{self.continent}"

        location = self.geolocator.geocode(search_query)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None

    def get_location_info(self, text):
        self.generate_location(text)
        self.parse_location()
        latitude, longitude = self.get_coordinates()
        return {
            "city": self.city,
            "country": self.country,
            "continent": self.continent,
            "latitude": latitude,
            "longitude": longitude
        }

# Example usage:
if __name__ == "__main__":
    tex2loc = Tex2loc(model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    text = "The Eiffel Tower is located in Paris, France."
    location_info = tex2loc.get_location_info(text)
    print(location_info)
    # Output: {'city': 'Paris', 'country': 'France', 'continent': 'Europe', 'latitude': 48.8588443, 'longitude': 2.2943506} 