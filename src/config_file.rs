//! Shared RON configuration file with one section per executable:
//!
//! ```ron
//! (
//!     fit: ( /* parameters of bin/fit */ ),
//!     centrality: ( /* parameters of bin/define-centrality */ ),
//! )
//! ```
//!
//! Each executable reads only its own section; the other sections must be
//! valid RON but are not interpreted.

use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::{self, DeserializeOwned, DeserializeSeed, IgnoredAny, MapAccess, Visitor};

use crate::error::{Error, Result};

/// Reads section `section` of the RON file at `path`.
pub fn read_section<T: DeserializeOwned>(path: &Path, section: &'static str) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
    parse_section(&text, section).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
}

/// Parses section `section` of a RON configuration. `Option` fields may be
/// written without `Some(...)`.
pub fn parse_section<T: DeserializeOwned>(
    text: &str,
    section: &'static str,
) -> std::result::Result<T, ron::error::SpannedError> {
    ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str_seed(
            text,
            Section {
                name: section,
                marker: PhantomData,
            },
        )
}

/// Expands a leading `~/` to `$HOME`.
pub fn expand_home(path: &Path) -> PathBuf {
    match (path.strip_prefix("~"), std::env::var_os("HOME")) {
        (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => path.to_path_buf(),
    }
}

struct Section<T> {
    name: &'static str,
    marker: PhantomData<T>,
}

impl<'de, T: DeserializeOwned> DeserializeSeed<'de> for Section<T> {
    type Value = T;

    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<T, D::Error> {
        /* the section names are not known in advance, the visitor accepts any */
        deserializer.deserialize_struct("Config", &[], self)
    }
}

impl<'de, T: DeserializeOwned> Visitor<'de> for Section<T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a configuration with a `{}` section", self.name)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<T, A::Error> {
        let mut value = None;
        let mut seen = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == self.name {
                if value.is_some() {
                    return Err(de::Error::custom(format!("duplicate section `{key}`")));
                }
                value = Some(map.next_value::<T>()?);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
            seen.push(key);
        }
        value.ok_or_else(|| {
            let found = if seen.is_empty() {
                "none".to_string()
            } else {
                seen.join(", ")
            };
            de::Error::custom(format!("missing section `{}` (found: {found})", self.name))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct A {
        x: i32,
        y: Option<u64>,
    }

    #[test]
    fn reads_own_section_and_skips_others() {
        let text = r#"(
            other: (anything: [1, 2, 3], nested: (z: "s")),
            a: (x: 1, y: 7),
            more: 5,
        )"#;
        assert_eq!(
            parse_section::<A>(text, "a").unwrap(),
            A { x: 1, y: Some(7) }
        );
    }

    #[test]
    fn errors() {
        let missing = parse_section::<A>("(b: (x: 1))", "a")
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("missing section `a`") && missing.contains('b'),
            "{missing}"
        );
        assert!(parse_section::<A>("(a: (x: 1), a: (x: 2))", "a").is_err());
        assert!(parse_section::<A>("(a: (x: 1, typo: 2))", "a").is_err());
        assert!(parse_section::<A>("(a: (x: 1), b: (unclosed)", "a").is_err());
    }

    #[test]
    fn home_expansion() {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        if let Some(home) = home {
            assert_eq!(expand_home(Path::new("~/a/b")), home.join("a/b"));
        }
        assert_eq!(expand_home(Path::new("/x/~/y")), PathBuf::from("/x/~/y"));
    }
}
