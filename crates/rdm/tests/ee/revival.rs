// Included by tests/ee_arbiter.rs, whose helpers it uses.
struct ReviewRevival {
    talk: Talk,
    reports: Arc<Mutex<Vec<(usize, bool)>>>,
}
impl Arbiter for ReviewRevival {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        if ledger.round == 0 {
            self.talk.more();
            Ok(finish())
        } else {
            Err(rdm::Error::Tool("synthetic unavailable arbiter".into()))
        }
    }
    fn acted(&self, ledger: &Ledger, _: &Ruling, applied: bool) {
        self.reports.lock().unwrap().push((ledger.round, applied));
    }
}
#[test]
fn reused_ruling_reports_every_recorded_attempts_disposition() {
    let (tool, talk) = in_step(&["v1", "v2"], DEADLINE);
    let reports = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded = events.clone();
    let got = Rdm::new(Signature::new("Review callback identity").noul("chatty", "Chatty?"))
        .model(model()).arbiter(ReviewRevival { talk, reports: reports.clone() }).tool(tool)
        .record(move |event: &rdm::record::Event| {
            if let rdm::record::Event::Arbitrated { round, .. } = event { recorded.lock().unwrap().push(*round); }
            Ok(())
        }).call(inputs! {}).unwrap();
    let arbitration = got.arbitration.unwrap();
    assert!(arbitration.answered && arbitration.used == ["chatty"]);
    assert_eq!(got.observations["chatty"], "v1");
    assert_eq!(arbitration.versions, [("chatty".to_owned(), 1)]);
    let events = events.lock().unwrap().clone();
    let reports = reports.lock().unwrap().clone();
    assert_eq!(events, [0, 1], "must reach a discarded paid ruling and then its failed successor");
    assert_eq!(reports.first(), Some(&(0, false)), "first ruling must really be discarded");
    eprintln!("recorded rounds={events:?}; acted callbacks={reports:?}; final accepted versions={:?}", arbitration.versions);
    let recorded: std::collections::BTreeSet<_> = events.into_iter().collect();
    let notified: std::collections::BTreeSet<_> = reports.into_iter().map(|(round, _)| round).collect();
    assert_eq!(recorded, notified, "An emitted successor ruling has no applied/discarded callback; the old discarded round was revived instead");
}
