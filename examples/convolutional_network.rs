use image::ImageReader;

fn main() -> Result<(), Box<dyn::Error>> {
    println!("test from a new project folder!")
    let img = ImageReader::open("C:\\Users\\Shaka\\Downloads\\archive\\mnist_png\\test\\4")?.decode()?;
    img.save("gradient.png")?;
    Ok(())
}