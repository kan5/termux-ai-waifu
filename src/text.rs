//! Text normalization for the TTS input.
//!
//! Silero TTS (`v5_5_ru`) filters input to its fixed symbol alphabet and
//! silently drops anything outside it — including ASCII digits 0-9. Before
//! sending text to TTS we therefore convert digit runs to Russian words
//! (nominative/citation form, which is the right default for standalone
//! numbers like "123" -> "сто двадцать три").

/// Replace every run of ASCII digits with its Russian word form.
pub fn normalize_digits(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let num_str: String = chars[i..j].iter().collect();
            match num_str.parse::<u64>() {
                Ok(n) => out.push_str(&number_to_words_ru(n)),
                Err(_) => out.push_str(&num_str), // too large to fit u64 — leave as-is
            }
            i = j;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Convert a non-negative integer to Russian words (nominative, citation form).
fn number_to_words_ru(n: u64) -> String {
    if n == 0 {
        return "ноль".to_string();
    }

    const UNITS: [&str; 20] = [
        "", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
        "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
        "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    ];
    const TENS: [&str; 10] = [
        "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
        "восемьдесят", "девяносто",
    ];
    const HUNDREDS: [&str; 10] = [
        "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
        "восемьсот", "девятьсот",
    ];

    fn under_1000(n: u64, fem: bool) -> String {
        let h = (n / 100) as usize;
        let rem = n % 100;
        let t = (rem / 10) as usize;
        let u = (rem % 10) as usize;
        let mut parts: Vec<&str> = Vec::new();
        if h > 0 {
            parts.push(HUNDREDS[h]);
        }
        if rem >= 20 {
            if t > 0 {
                parts.push(TENS[t]);
            }
            if u > 0 {
                parts.push(match (fem, u) {
                    (true, 1) => "одна",
                    (true, 2) => "две",
                    _ => UNITS[u],
                });
            }
        } else if rem > 0 {
            parts.push(match (fem, rem as usize) {
                (true, 1) => "одна",
                (true, 2) => "две",
                _ => UNITS[rem as usize],
            });
        }
        parts.join(" ")
    }

    fn plural_scale(n: u64, one: &str, few: &str, many: &str) -> String {
        let rem10 = n % 10;
        let rem100 = n % 100;
        let form = if rem10 == 1 && rem100 != 11 {
            one
        } else if (2..=4).contains(&rem10) && !(12..=14).contains(&rem100) {
            few
        } else {
            many
        };
        form.to_string()
    }

    let billions = n / 1_000_000_000;
    let millions = (n / 1_000_000) % 1000;
    let thousands = (n / 1000) % 1000;
    let rest = n % 1000;

    let mut parts: Vec<String> = Vec::new();
    if billions > 0 {
        parts.push(format!(
            "{} {}",
            under_1000(billions, false),
            plural_scale(billions, "миллиард", "миллиарда", "миллиардов")
        ));
    }
    if millions > 0 {
        parts.push(format!(
            "{} {}",
            under_1000(millions, false),
            plural_scale(millions, "миллион", "миллиона", "миллионов")
        ));
    }
    if thousands > 0 {
        parts.push(format!(
            "{} {}",
            under_1000(thousands, true),
            plural_scale(thousands, "тысяча", "тысячи", "тысяч")
        ));
    }
    if rest > 0 {
        parts.push(under_1000(rest, false));
    }
    parts.join(" ")
}

/// Strip Qwen3's `<think>…</think>` reasoning blocks from a text fragment.
/// `in_think` persists across calls so blocks spanning chunk boundaries are
/// handled correctly.
pub fn filter_think(chunk: &str, in_think: &mut bool) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut out = String::new();
    let mut rest = chunk;
    loop {
        if *in_think {
            match rest.find(CLOSE) {
                Some(idx) => {
                    *in_think = false;
                    rest = &rest[idx + CLOSE.len()..];
                }
                None => break, // still inside a think block — drop the rest
            }
        } else {
            match rest.find(OPEN) {
                Some(idx) => {
                    out.push_str(&rest[..idx]);
                    *in_think = true;
                    rest = &rest[idx + OPEN.len()..];
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_to_words() {
        assert_eq!(normalize_digits("мне 3 года"), "мне три года");
        assert_eq!(normalize_digits("123"), "сто двадцать три");
        assert_eq!(normalize_digits("42"), "сорок два");
        assert_eq!(normalize_digits("0"), "ноль");
        assert_eq!(normalize_digits("2024 год"), "две тысячи двадцать четыре год");
        assert_eq!(normalize_digits("1000"), "одна тысяча");
        assert_eq!(normalize_digits("2000"), "две тысячи");
        assert_eq!(normalize_digits("1000000"), "один миллион");
        assert_eq!(normalize_digits("текст без цифр"), "текст без цифр");
    }

    #[test]
    fn think_blocks_stripped() {
        let mut in_think = false;
        assert_eq!(filter_think("<think>hmm</think>answer", &mut in_think), "answer");
        assert!(!in_think);
    }
}
