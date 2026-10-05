package com.example.eventtracker.controller;

import com.example.eventtracker.entity.Admin;
import com.example.eventtracker.repository.AdminRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/admin")
public class AdminController {

    @Autowired
    private AdminRepository adminRepository;

    @PostMapping("/setup")
    public ResponseEntity<?> setupAdmin(@RequestBody Admin admin) {
        if (adminRepository.findByUsername(admin.getUsername()).isPresent()) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body("Username already exists");
        }
        Admin savedAdmin = adminRepository.save(admin);
        return ResponseEntity.ok(savedAdmin);
    }

    @PostMapping("/login")
    public ResponseEntity<?> loginAdmin(@RequestBody Admin loginRequest) {
        Optional<Admin> adminOpt = adminRepository.findByUsername(loginRequest.getUsername());
        if (adminOpt.isPresent() && adminOpt.get().getPassword().equals(loginRequest.getPassword())) {
            return ResponseEntity.ok().body(Map.of("message", "Login successful", "success", true));
        }
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("message", "Invalid credentials", "success", false));
    }
}
