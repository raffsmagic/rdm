//! What a ruling saw is kept and read back, and a record kept before it still reads.
use rdm::arbitrate::Ruling;
use rdm::record::Event;
#[test]
fn old_arbitrated_events_without_saw_still_roundtrip_without_a_new_field() {
    let original = Event::Arbitrated {
        run: 1,
        index: 0,
        round: 3,
        ruling: Ruling::default(),
        error: None,
        saw: vec![],
    };
    let json = serde_json::to_value(&original).unwrap();
    assert!(json.get("saw").is_none());
    let Event::Arbitrated { saw, round, .. } = serde_json::from_value::<Event>(json).unwrap() else {
        panic!("wrong event")
    };
    assert!(saw.is_empty());
    assert_eq!(round, 3);
}
#[test]
fn saw_preserves_both_returned_and_partial_identities_on_roundtrip() {
    let versions = vec![("read".into(), 0), ("writer".into(), 7)];
    let original = Event::Arbitrated {
        run: 1,
        index: 0,
        round: 3,
        ruling: Ruling::default(),
        error: None,
        saw: versions.clone(),
    };
    let Event::Arbitrated { saw, .. } = serde_json::from_value::<Event>(serde_json::to_value(&original).unwrap()).unwrap() else {
        panic!("wrong event")
    };
    assert_eq!(saw, versions);
}
