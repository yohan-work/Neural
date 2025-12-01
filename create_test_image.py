from PIL import Image, ImageDraw, ImageFont
import random

def create_dummy_digit(filename='test_digit.png', digit='3'):
    # Create a white image (simulating paper)
    img = Image.new('L', (300, 300), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw a black digit (simulating ink)
    # Since we might not have a font, we'll draw lines to simulate '3' or just use default font
    # Let's try to draw a thick '3' manually to be safe from font issues, or use a large default font
    
    # Simple manual drawing of '3'
    # Top bar
    draw.line([(50, 50), (250, 50)], fill=0, width=20)
    # Middle bar
    draw.line([(50, 150), (250, 150)], fill=0, width=20)
    # Bottom bar
    draw.line([(50, 250), (250, 250)], fill=0, width=20)
    # Right side vertical
    draw.line([(250, 50), (250, 250)], fill=0, width=20)
    
    img.save(filename)
    print(f"Created dummy image: {filename}")

if __name__ == "__main__":
    create_dummy_digit()
