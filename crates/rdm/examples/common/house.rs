//! The house: the simulated smart home the cells share. It is a value the tools change and
//! every request shows as it is then, which in Rust is a `Shared` given as a live input.

use rdm::{Json, Shared};
use serde_json::Map;

use super::show::show_house;

/// The house: each device and the state it is in, in the order they were given.
///
/// `house.lock()["door"]` reads a device, `house.get()` is a copy as it is now (`dict(house)`
/// in Python), `house.set(start.clone())` puts it back (`house.update(start)`), and
/// `inputs! { "house" => &house }` shows every request the house as it is then.
pub type House = Shared<Map<String, Json>>;

/// A house that starts as an object says: `house(json!({"tv": "off", "door": "open"}))`.
pub fn house(start: Json) -> House {
    match start {
        Json::Object(devices) => Shared::new(devices),
        _ => Shared::new(Map::new()),
    }
}

/// Set one device to one state. The house pushes back: an open door cannot be locked.
pub fn switch(house: &House, device: &str, to: &str) -> String {
    let done = {
        let mut house = house.lock();
        if device == "lock" && to == "locked" && house.get("door").is_some_and(|door| door == "open") {
            "lock: refused, the door is open".to_owned()
        } else {
            house.insert(device.to_owned(), to.into());
            format!("{device}: now {to}")
        }
    };
    show_house(house, &done); // redraws the house on the right
    done
}

/// How warm it is, in degrees.
pub fn heat(house: &House) -> i64 {
    house.lock().get("heat").and_then(Json::as_i64).unwrap_or_default()
}

/// Make it one degree cooler.
pub fn cooler(house: &House) -> String {
    warmed(house, -1)
}

/// Make it one degree warmer.
pub fn warmer(house: &House) -> String {
    warmed(house, 1)
}

fn warmed(house: &House, by: i64) -> String {
    let now = heat(house) + by;
    house.lock().insert("heat".to_owned(), now.into());
    let done = format!("heat: now {now}");
    show_house(house, &done);
    done
}
