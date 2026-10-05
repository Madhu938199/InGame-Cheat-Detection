package com.example.eventtracker.controller;

import com.example.eventtracker.entity.Participant;
import com.example.eventtracker.repository.ParticipantRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/participants")
public class ParticipantController {

    private static final Logger logger = LoggerFactory.getLogger(ParticipantController.class);

    @Autowired
    private ParticipantRepository participantRepository;

    @PostMapping("/register")
    public ResponseEntity<Participant> registerParticipant(@RequestBody Participant participant) {
        Participant savedParticipant = participantRepository.save(participant);
        logger.info("✅ SUCCESS: Participant Registered! Name: {}, UUID: {}", 
                    savedParticipant.getFullName(), savedParticipant.getId());
        return ResponseEntity.ok(savedParticipant);
    }

    @GetMapping
    public ResponseEntity<List<Participant>> getAllParticipants() {
        return ResponseEntity.ok(participantRepository.findAll());
    }

    @PatchMapping("/{id}/attendance")
    public ResponseEntity<?> toggleAttendance(@PathVariable String id, @RequestBody Map<String, Boolean> update) {
        return participantRepository.findById(id).map(participant -> {
            Boolean isAttended = update.getOrDefault("isAttended", participant.getIsAttended());
            participant.setIsAttended(isAttended);
            participantRepository.save(participant);
            return ResponseEntity.ok(participant);
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteParticipant(@PathVariable String id) {
        return participantRepository.findById(id).map(participant -> {
            participantRepository.delete(participant);
            return ResponseEntity.ok().build();
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }
}
