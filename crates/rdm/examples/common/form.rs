//! The form above a cell. On the site a form rewrites the lines of the cell that assign its
//! inputs (`task = "..."`), and offers presets for each. Here the same inputs come from the
//! command line, each defaulting to the cell's own value, and `--help` lists the presets.
//!
//! ```text
//! cargo run -p rdm --example act                                   the cell as it is written
//! cargo run -p rdm --example act -- "Please shut the door."        the first input, by position
//! cargo run -p rdm --example act -- --task 3                       a preset, by its number
//! cargo run -p rdm --example act -- --task "Open up." --house '{"tv": "on", "door": "closed"}'
//! cargo run -p rdm --example act -- --help
//! ```

use rdm::Json;

use super::house::{House, house};

enum Kind {
    /// One text, with the presets the form offers and a short name for each where it has one.
    Text { presets: Vec<String>, short: Vec<String> },
    /// Several texts, one to a line.
    Lines,
    /// Anything written as JSON: the house.
    Json,
}

struct Field {
    name: String,
    label: String,
    kind: Kind,
    value: Json,
}

/// The inputs of an example: what `index.js` says of its cell, and the values the cell gives
/// them. Declared in the order of the form, then read from the command line with [`Form::read`].
pub struct Form {
    name: String,
    label: String,
    blurb: String,
    fields: Vec<Field>,
}

impl Form {
    /// The form of the example `name`, with the cell's label and blurb from `index.js`.
    pub fn new(name: &str, label: &str, blurb: &str) -> Self {
        Form {
            name: name.to_owned(),
            label: label.to_owned(),
            blurb: blurb.to_owned(),
            fields: Vec::new(),
        }
    }

    fn field(mut self, name: &str, label: &str, kind: Kind, value: Json) -> Self {
        self.fields.push(Field {
            name: name.to_owned(),
            label: label.to_owned(),
            kind,
            value,
        });
        self
    }

    /// A text, with its presets. It starts as the first, which is what the cell assigns.
    pub fn text(self, name: &str, label: &str, presets: &[&str]) -> Self {
        let presets: Vec<String> = presets.iter().map(|preset| preset.to_string()).collect();
        let value = presets.first().cloned().unwrap_or_default().into();
        let short = Vec::new();
        self.field(name, label, Kind::Text { presets, short }, value)
    }

    /// A text whose presets each have a short name, as a `textarea` of the site has: `(short, preset)`.
    pub fn text_named(self, name: &str, label: &str, presets: &[(&str, &str)]) -> Self {
        let short = presets.iter().map(|(short, _)| short.to_string()).collect();
        let presets: Vec<String> = presets.iter().map(|(_, preset)| preset.to_string()).collect();
        let value = presets.first().cloned().unwrap_or_default().into();
        self.field(name, label, Kind::Text { presets, short }, value)
    }

    /// Several texts, one to a line on the site, starting as the cell lists them.
    pub fn lines(self, name: &str, label: &str, start: &[&str]) -> Self {
        self.field(name, label, Kind::Lines, start.to_vec().into())
    }

    /// Anything written as JSON, starting as the cell writes it.
    pub fn json(self, name: &str, label: &str, start: Json) -> Self {
        self.field(name, label, Kind::Json, start)
    }

    /// Where the house starts: the input `house` of the smart home cells.
    pub fn house(self, start: Json) -> Self {
        self.json("house", "Where the house starts", start)
    }

    /// The form as the command line fills it in. `--help` prints what it takes and ends the
    /// program; an argument it does not take ends it with why.
    pub fn read(self) -> Self {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        if arguments.iter().any(|argument| argument == "--help" || argument == "-h") {
            print!("{}", self.help());
            std::process::exit(0);
        }
        match self.filled(arguments) {
            Ok(form) => form,
            Err((form, why)) => {
                eprintln!("{why}\n\n{}", form.usage());
                std::process::exit(2);
            }
        }
    }

    /// The form filled in from these arguments, or the form and why they do not fit it.
    pub fn filled(mut self, arguments: Vec<String>) -> Result<Self, (Self, String)> {
        match self.fill(arguments) {
            Ok(()) => Ok(self),
            Err(why) => Err((self, why)),
        }
    }

    fn fill(&mut self, arguments: Vec<String>) -> Result<(), String> {
        // What was given for each input, in the order it was given.
        let mut given: Vec<Vec<String>> = vec![Vec::new(); self.fields.len()];
        let mut loose = Vec::new();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let Some(flag) = argument.strip_prefix("--").filter(|flag| !flag.is_empty()) else {
                loose.push(argument);
                continue;
            };
            if flag == "trace" {
                continue;
            }
            let (name, value) = match flag.split_once('=') {
                Some((name, value)) => (name, Some(value.to_owned())),
                None => (flag, None),
            };
            let at = self.fields.iter().position(|field| field.name == name);
            let at = at.ok_or_else(|| format!("The example `{}` takes no --{name}.", self.name))?;
            let value = value
                .or_else(|| arguments.next())
                .ok_or_else(|| format!("--{name} needs a value."))?;
            given[at].push(value);
        }
        // What was given without a name is for the inputs that are text, in the order of the form.
        // The last of them that takes several takes all that is left.
        let mut loose = loose.into_iter().peekable();
        for (at, field) in self.fields.iter().enumerate() {
            match field.kind {
                Kind::Text { .. } if given[at].is_empty() => given[at].extend(loose.next()),
                Kind::Lines if given[at].is_empty() => given[at].extend(loose.by_ref()),
                _ => {}
            }
        }
        if let Some(extra) = loose.next() {
            return Err(format!("The example `{}` has no input left for {extra:?}.", self.name));
        }
        for (field, given) in self.fields.iter_mut().zip(given) {
            let Some(last) = given.last() else { continue };
            field.value = match &field.kind {
                Kind::Text { presets, .. } => match last.parse::<usize>() {
                    Ok(number) if (1..=presets.len()).contains(&number) => presets[number - 1].clone().into(),
                    _ => last.clone().into(),
                },
                Kind::Lines => {
                    let lines = given.iter().flat_map(|each| each.lines()).map(str::trim);
                    lines.filter(|line| !line.is_empty()).collect::<Vec<_>>().into()
                }
                Kind::Json => serde_json::from_str(last).map_err(|error| format!("--{} is not JSON: {error}", field.name))?,
            };
        }
        Ok(())
    }

    fn value(&self, name: &str) -> &Json {
        let field = self.fields.iter().find(|field| field.name == name);
        &field
            .unwrap_or_else(|| panic!("the form of `{}` has no input `{name}`", self.name))
            .value
    }

    /// The text an input is set to.
    pub fn text_of(&self, name: &str) -> String {
        self.value(name).as_str().unwrap_or_default().to_owned()
    }

    /// The texts an input of several is set to.
    pub fn lines_of(&self, name: &str) -> Vec<String> {
        let lines = self.value(name).as_array().into_iter().flatten();
        lines.filter_map(|line| line.as_str().map(str::to_owned)).collect()
    }

    /// What an input written as JSON is set to.
    pub fn json_of(&self, name: &str) -> Json {
        self.value(name).clone()
    }

    /// The house, as the input `house` starts it.
    pub fn house_of(&self) -> House {
        house(self.json_of("house"))
    }

    fn usage(&self) -> String {
        let each = self.fields.iter().map(|field| match field.kind {
            Kind::Text { ref presets, .. } if !presets.is_empty() => format!(" [--{} <text, or 1-{}>]", field.name, presets.len()),
            Kind::Text { .. } => format!(" [--{} <text>]", field.name),
            Kind::Lines => format!(" [--{} <text>]...", field.name),
            Kind::Json => format!(" [--{} <JSON>]", field.name),
        });
        format!(
            "cargo run -p rdm --example {} --{} [--trace]\nMore with --help.",
            self.name,
            each.collect::<String>()
        )
    }

    /// What `--help` prints: the cell's label and blurb, each input with its presets, and how the models are chosen.
    pub fn help(&self) -> String {
        let mut help = format!("{}: {}\n{}\n", self.name, self.label, self.blurb);
        for field in &self.fields {
            help.push_str(&format!("\n--{}   {}\n", field.name, field.label));
            match &field.kind {
                Kind::Text { presets, short } => {
                    for (at, preset) in presets.iter().enumerate() {
                        let short = short.get(at).map(|short| format!("{short}: ")).unwrap_or_default();
                        help.push_str(&format!("    {}  {short}{preset}\n", at + 1));
                    }
                    if !presets.is_empty() {
                        help.push_str("    A number picks one of these, and anything else is the text itself. It starts as 1.\n");
                    }
                }
                Kind::Lines => {
                    for line in field.value.as_array().into_iter().flatten() {
                        help.push_str(&format!("    {}\n", line.as_str().unwrap_or_default()));
                    }
                    help.push_str("    Given once for each line; it starts as these.\n");
                }
                Kind::Json => help.push_str(&format!("    JSON; it starts as {}\n", rdm::as_observation(&field.value))),
            }
        }
        help.push_str(&format!(
            "\n--trace   Show the run as the site does: every request, under the one whose tool made it (or RDM_TRACE=1).\n\
             \nAn input can also be given without its name, in the order above:\n    cargo run -p rdm --example {} -- \"...\"\n\
             \nThe models are chosen by the environment (crates/rdm/examples/README.md):\n    \
             RDM_DECISION_MODEL   standin | typesafe | openrouter | local | <url> | gateway:<registry.json> | lm\n    \
             RDM_LANGUAGE_MODEL   standin | <an lm15 model, as openrouter:typesafe/jev-router> | <url>\n\
             With neither set and no key in the environment, both are stand-ins and nothing leaves this machine.\n",
            self.name
        ));
        help
    }
}
