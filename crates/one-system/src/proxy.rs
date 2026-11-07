//! Which proxy a server is reached through, if any, as the environment says.
//!
//! The rules are those of Go's `http.ProxyFromEnvironment`, which the Go gateway uses, since
//! the HTTP client here would otherwise take one proxy for every server alike:
//!
//! - a server on this machine (`localhost`, or a loopback address) is never proxied;
//! - an `https` server goes by `HTTPS_PROXY` and an `http` one by `HTTP_PROXY`, or by the
//!   same names in lower case;
//! - a host that `NO_PROXY` (or `no_proxy`) names is not proxied: `*` for every host, a
//!   domain for itself and what is under it, `.domain` for what is under it only, an address,
//!   or a block of addresses (`10.0.0.0/8`), any of them with a port to mean that port alone.
//!
//! One thing is added to Go's rules: `ALL_PROXY`, for a scheme that has no variable of its
//! own set, as curl has it. Go reads no such variable.

use std::net::IpAddr;

/// The parts of a base URL that say where its server is.
struct Server {
    /// `http` or `https`, in lower case. Empty for a URL that names no scheme.
    scheme: String,
    /// The host in lower case, an IPv6 address without its brackets.
    host: String,
    /// The port, as written or as the scheme implies. Empty when neither says.
    port: String,
}

impl Server {
    fn of(base_url: &str) -> Self {
        let (scheme, rest) = match base_url.split_once("://") {
            Some((scheme, rest)) => (scheme.to_ascii_lowercase(), rest),
            None => (String::new(), base_url),
        };
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        let authority = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
        let (host, port) = match authority.strip_prefix('[') {
            Some(bracketed) => {
                let (host, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
                (host, after.strip_prefix(':').unwrap_or_default())
            }
            None => authority.rsplit_once(':').unwrap_or((authority, "")),
        };
        let implied = match scheme.as_str() {
            "http" => "80",
            "https" => "443",
            _ => "",
        };
        Self {
            host: host.to_ascii_lowercase(),
            port: if port.is_empty() { implied } else { port }.to_string(),
            scheme,
        }
    }

    /// The host as an address, if it is one. An IPv4 address written as IPv6 is IPv4.
    fn address(&self) -> Option<IpAddr> {
        self.host.parse::<IpAddr>().ok().map(|address| address.to_canonical())
    }

    /// Whether it is on this machine.
    fn local(&self) -> bool {
        self.host == "localhost" || self.address().is_some_and(|address| address.is_loopback())
    }

    /// Whether a list of exemptions, as `NO_PROXY` holds one, names it.
    fn exempt(&self, list: &str) -> bool {
        let address = self.address();
        list.split(',').any(|entry| {
            let entry = entry.trim().to_ascii_lowercase();
            if entry.is_empty() {
                return false;
            }
            if entry == "*" {
                return true;
            }
            // A block of addresses.
            if let Some((network, bits)) = entry.split_once('/') {
                return match (network.parse::<IpAddr>(), bits.parse::<u32>(), address) {
                    (Ok(network), Ok(bits), Some(address)) => within(network, bits, address),
                    _ => false,
                };
            }
            // An address or a domain, alone or with a port: `10.0.0.5`, `[::1]:8080`, `example.com:443`.
            let (host, port) = match entry.strip_prefix('[') {
                Some(bracketed) => {
                    let (host, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
                    (host, after.strip_prefix(':').unwrap_or_default())
                }
                None if entry.parse::<IpAddr>().is_ok() => (entry.as_str(), ""),
                None => entry.rsplit_once(':').unwrap_or((entry.as_str(), "")),
            };
            if host.is_empty() || !(port.is_empty() || port == self.port) {
                return false;
            }
            if let Ok(named) = host.parse::<IpAddr>() {
                return address == Some(named.to_canonical());
            }
            // `example.com` is itself and what is under it; `.example.com` and
            // `*.example.com` are only what is under it.
            let host = host.strip_prefix('*').filter(|rest| rest.starts_with('.')).unwrap_or(host);
            match host.strip_prefix('.') {
                Some(domain) => self.host.strip_suffix(domain).is_some_and(|under| under.ends_with('.')),
                None => self.host == host || self.host.strip_suffix(host).is_some_and(|under| under.ends_with('.')),
            }
        })
    }
}

/// Whether an address is inside a block of them: the first `bits` bits are the network's.
fn within(network: IpAddr, bits: u32, address: IpAddr) -> bool {
    let masked = |value: u128, width: u32| if bits == 0 { 0 } else { value >> (width - bits.min(width)) };
    match (network, address) {
        (IpAddr::V4(network), IpAddr::V4(address)) => {
            bits <= 32 && masked(u32::from(network).into(), 32) == masked(u32::from(address).into(), 32)
        }
        (IpAddr::V6(network), IpAddr::V6(address)) => bits <= 128 && masked(network.into(), 128) == masked(address.into(), 128),
        _ => false,
    }
}

/// The proxy the server at a base URL is to be reached through, or none. `variable` reads
/// one variable of the environment.
pub(crate) fn for_server(base_url: &str, variable: impl Fn(&str) -> Option<String>) -> Option<String> {
    let server = Server::of(base_url);
    if server.local() {
        return None;
    }
    // The first of these that holds anything, as Go reads them: upper case before lower.
    let first = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| variable(name).filter(|value| !value.trim().is_empty()))
    };
    let own = match server.scheme.as_str() {
        "https" => first(&["HTTPS_PROXY", "https_proxy"]),
        // A program run as CGI is handed a caller's `Proxy` header as `HTTP_PROXY`. It is not
        // believed there, and `http_proxy` is not what such a header becomes.
        "http" if first(&["REQUEST_METHOD"]).is_some() => first(&["http_proxy"]),
        "http" => first(&["HTTP_PROXY", "http_proxy"]),
        _ => return None,
    };
    let proxy = own.or_else(|| first(&["ALL_PROXY", "all_proxy"]))?;
    if first(&["NO_PROXY", "no_proxy"]).is_some_and(|list| server.exempt(&list)) {
        return None;
    }
    Some(proxy.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROXY: &str = "http://proxy.corp.example:3128";

    /// The proxy for a server, in an environment of these variables and no others.
    fn through(base_url: &str, environment: &[(&str, &str)]) -> Option<String> {
        for_server(base_url, |name| {
            environment
                .iter()
                .find(|(variable, _)| *variable == name)
                .map(|(_, value)| value.to_string())
        })
    }

    #[test]
    fn a_server_on_this_machine_is_never_proxied() {
        let everything = [("HTTP_PROXY", PROXY), ("HTTPS_PROXY", PROXY), ("ALL_PROXY", PROXY)];
        for local in [
            "http://127.0.0.1:8091",
            "http://127.0.0.1",
            "http://127.8.9.10:1/v1",
            "http://localhost:8080",
            "HTTP://LOCALHOST:8080",
            "https://localhost",
            "http://[::1]:8080",
            "http://[::ffff:127.0.0.1]:8091",
            "https://user:pass@127.0.0.1:8443/path",
        ] {
            assert_eq!(through(local, &everything), None, "{local}");
        }
        // A name that only begins like this machine's is another machine.
        assert_eq!(through("https://localhost.example.com", &everything).as_deref(), Some(PROXY));
        assert_eq!(through("https://10.0.0.5", &everything).as_deref(), Some(PROXY));
    }

    #[test]
    fn each_scheme_has_its_own_variable() {
        let other = "http://other.example:8080";
        assert_eq!(
            through("https://api.typesafe.ai", &[("HTTPS_PROXY", PROXY), ("HTTP_PROXY", other)]).as_deref(),
            Some(PROXY)
        );
        assert_eq!(
            through("http://example.com", &[("HTTPS_PROXY", other), ("HTTP_PROXY", PROXY)]).as_deref(),
            Some(PROXY)
        );
        // The other scheme's proxy is not this one's.
        assert_eq!(through("https://api.typesafe.ai", &[("HTTP_PROXY", PROXY)]), None);
        assert_eq!(through("http://example.com", &[("HTTPS_PROXY", PROXY)]), None);
        // Lower case is read, after upper case, and a variable that is empty is one that is not set.
        assert_eq!(
            through("https://api.typesafe.ai", &[("https_proxy", PROXY)]).as_deref(),
            Some(PROXY)
        );
        assert_eq!(
            through("https://api.typesafe.ai", &[("HTTPS_PROXY", PROXY), ("https_proxy", other)]).as_deref(),
            Some(PROXY)
        );
        assert_eq!(
            through("https://api.typesafe.ai", &[("HTTPS_PROXY", " "), ("https_proxy", PROXY)]).as_deref(),
            Some(PROXY)
        );
        assert_eq!(
            through("HTTPS://api.typesafe.ai", &[("HTTPS_PROXY", PROXY)]).as_deref(),
            Some(PROXY)
        );
        // A proxy for everything stands in for a scheme that has none of its own.
        assert_eq!(through("https://api.typesafe.ai", &[("ALL_PROXY", PROXY)]).as_deref(), Some(PROXY));
        assert_eq!(
            through("https://api.typesafe.ai", &[("ALL_PROXY", other), ("HTTPS_PROXY", PROXY)]).as_deref(),
            Some(PROXY)
        );
        assert_eq!(
            through("http://example.com", &[("all_proxy", PROXY), ("HTTPS_PROXY", other)]).as_deref(),
            Some(PROXY)
        );
        // Nothing set, or a URL that is not one.
        assert_eq!(through("https://api.typesafe.ai", &[]), None);
        assert_eq!(through("api.typesafe.ai", &[("HTTPS_PROXY", PROXY), ("ALL_PROXY", PROXY)]), None);
        // What a CGI server was sent as a header is not a proxy.
        assert_eq!(
            through("http://example.com", &[("HTTP_PROXY", PROXY), ("REQUEST_METHOD", "GET")]),
            None
        );
        assert_eq!(
            through("http://example.com", &[("http_proxy", PROXY), ("REQUEST_METHOD", "GET")]).as_deref(),
            Some(PROXY)
        );
    }

    #[test]
    fn hosts_the_environment_exempts() {
        let exempt =
            |base_url: &str, list: &str| through(base_url, &[("HTTPS_PROXY", PROXY), ("HTTP_PROXY", PROXY), ("NO_PROXY", list)]).is_none();
        let hosted = "https://api.typesafe.ai";
        assert!(exempt(hosted, "*"));
        assert!(exempt(hosted, "example.com, * "));
        assert!(exempt(hosted, "api.typesafe.ai"));
        assert!(exempt(hosted, "API.TypeSafe.ai"));
        assert!(exempt(hosted, "internal.example,typesafe.ai"));
        assert!(exempt(hosted, ".typesafe.ai"));
        assert!(exempt(hosted, "*.typesafe.ai"));
        assert!(exempt(hosted, "typesafe.ai:443"));
        assert!(!exempt(hosted, "typesafe.ai:8443"));
        assert!(!exempt(hosted, ""));
        assert!(!exempt(hosted, " , "));
        assert!(!exempt(hosted, "safe.ai"));
        assert!(!exempt(hosted, "api.typesafe.ai.evil.example"));
        assert!(!exempt(hosted, "other.example"));
        // What is under a domain, and not the domain.
        assert!(!exempt("https://typesafe.ai", ".typesafe.ai"));
        assert!(!exempt("https://typesafe.ai", "*.typesafe.ai"));
        assert!(exempt("https://typesafe.ai", "typesafe.ai"));
        // Addresses, and blocks of them.
        assert!(exempt("https://10.1.2.3:8443", "10.1.2.3"));
        assert!(exempt("https://10.1.2.3:8443", "10.1.2.3:8443"));
        assert!(!exempt("https://10.1.2.3:8443", "10.1.2.3:443"));
        assert!(!exempt("https://10.1.2.3", "10.1.2.4"));
        assert!(exempt("https://10.1.2.3", "10.0.0.0/8"));
        assert!(exempt("https://10.1.2.3", "0.0.0.0/0"));
        assert!(!exempt("https://11.1.2.3", "10.0.0.0/8"));
        assert!(!exempt("https://10.1.2.3", "10.1.2.4/32"));
        assert!(!exempt("https://10.1.2.3", "10.0.0.0/99"));
        assert!(exempt("https://[2001:db8::7]", "2001:db8::/32"));
        assert!(exempt("https://[2001:db8::7]:8443", "[2001:db8::7]:8443"));
        assert!(exempt("https://[2001:db8::7]", "2001:db8::7"));
        assert!(!exempt("https://[2001:db9::7]", "2001:db8::/32"));
        assert!(!exempt("https://[2001:db8::7]", "10.0.0.0/8"));
        // A name is not looked up to see whether its address is exempt.
        assert!(!exempt(hosted, "10.0.0.0/8"));
        // The lower-case list is read when the upper-case one is not set.
        assert_eq!(through(hosted, &[("HTTPS_PROXY", PROXY), ("no_proxy", "typesafe.ai")]), None);
    }

    #[test]
    fn where_a_server_is() {
        let of = |base_url: &str| {
            let server = Server::of(base_url);
            (server.scheme, server.host, server.port)
        };
        let parts = |scheme: &str, host: &str, port: &str| (scheme.to_string(), host.to_string(), port.to_string());
        assert_eq!(of("https://api.typesafe.ai"), parts("https", "api.typesafe.ai", "443"));
        assert_eq!(of("http://127.0.0.1:8091/v1"), parts("http", "127.0.0.1", "8091"));
        assert_eq!(of("https://openrouter.ai/api"), parts("https", "openrouter.ai", "443"));
        assert_eq!(of("http://[::1]:8080"), parts("http", "::1", "8080"));
        assert_eq!(of("http://[::1]"), parts("http", "::1", "80"));
        assert_eq!(of("https://u:p@Example.COM:8443?x#y"), parts("https", "example.com", "8443"));
        assert_eq!(of("example.com"), parts("", "example.com", ""));
    }
}
